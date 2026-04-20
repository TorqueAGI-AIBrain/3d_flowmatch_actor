# inference_node.py: ROS2 node for real-time VLA policy inference on xArm.
# inference_node.py: Subscribes to cameras + EEF, runs 3DFA model, publishes target pose.

"""
ROS2 inference node for 3D FlowMatch Actor policy deployment.

Subscribes to:
    - Front camera RGB + depth (Azure Kinect)
    - Wrist camera RGB + depth (RealSense D435i)
    - EEF pose (from xarm_ros2)
    - Gripper state

Publishes:
    - Target EEF pose (geometry_msgs/PoseStamped)

Usage:
    python -m inference.inference_node --config configs/inference.yaml

    # With instruction override:
    python -m inference.inference_node --config configs/inference.yaml \
        --instruction "place the object in the toolbox"
"""

import argparse
import collections
import threading
import time

import cv2
import numpy as np
import torch

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float64, String
    from cv_bridge import CvBridge
    from tf2_ros import Buffer, TransformListener
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

from utils.config import load_yaml_config, load_config_nested
from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class


IM_SIZE = 256


def parse_arguments():
    parser = argparse.ArgumentParser(description="3DFA inference node")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to inference YAML config")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override default instruction text")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device for inference")
    return parser.parse_args()


def load_model(config, device):
    """Load trained 3DFA model from checkpoint."""
    model_cls = fetch_model_class(config['model_type'])

    model = model_cls(
        backbone=config['backbone'],
        finetune_backbone=False,
        finetune_text_encoder=False,
        num_vis_instr_attn_layers=config['num_vis_instr_attn_layers'],
        fps_subsampling_factor=config['fps_subsampling_factor'],
        embedding_dim=config['embedding_dim'],
        num_attn_heads=config['num_attn_heads'],
        nhist=config['num_history'],
        nhand=2 if config.get('bimanual', False) else 1,
        num_shared_attn_layers=config['num_shared_attn_layers'],
        relative=config.get('relative_action', False),
        rotation_format=config.get('rotation_format', 'quat_xyzw'),
        denoise_timesteps=config['denoise_timesteps'],
        denoise_model=config['denoise_model'],
        lv2_batch_size=1,
    )

    # Load checkpoint
    ckpt_path = config['checkpoint']
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Handle DDP-wrapped keys (module. prefix)
    weights = ckpt["weight"]
    cleaned = {}
    for k, v in weights.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[key] = v
    model.load_state_dict(cleaned, strict=False)

    # Load workspace normalizer if present
    if "weight" in ckpt:
        for k, v in ckpt["weight"].items():
            if "workspace_normalizer" in k:
                key = k.replace("module.", "")
                parts = key.split(".")
                obj = model
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                setattr(obj, parts[-1], torch.nn.Parameter(v, requires_grad=False))

    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def preprocess_rgb(img_bgr, im_size=IM_SIZE):
    """Convert BGR uint8 image to (1, 1, 3, H, W) float tensor in [0, 1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (im_size, im_size))
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def preprocess_depth(depth_raw, im_size=IM_SIZE, depth_scale=1000.0):
    """Convert raw depth to (1, 1, H, W) float tensor in meters."""
    depth_m = depth_raw.astype(np.float32) / depth_scale
    depth_m = cv2.resize(depth_m, (im_size, im_size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(depth_m).float()


def depth_to_pointcloud(depth, intrinsics, extrinsics):
    """
    Convert depth image to world-frame point cloud.

    Args:
        depth: (H, W) tensor in meters
        intrinsics: (3, 3) tensor
        extrinsics: (4, 4) tensor cam-to-world

    Returns:
        pcd: (3, H, W) tensor of world coordinates
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = torch.arange(W, dtype=depth.dtype, device=depth.device)
    v = torch.arange(H, dtype=depth.dtype, device=depth.device)
    u, v = torch.meshgrid(u, v, indexing='xy')

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Camera-frame points (4, H*W)
    ones = torch.ones_like(z)
    pts_cam = torch.stack([x, y, z, ones], dim=0).reshape(4, -1)

    # Transform to world frame
    pts_world = extrinsics @ pts_cam  # (4, H*W)
    return pts_world[:3].reshape(3, H, W)


class PolicyInferenceNode(Node):
    """ROS2 node that runs 3DFA policy and publishes target EEF poses."""

    def __init__(self, model, config, device, instruction):
        super().__init__('policy_inference')

        self.model = model
        self.config = config
        self.device = device
        self.instruction = instruction
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.im_size = config.get('image_size', IM_SIZE)
        self.num_history = config['num_history']
        self.nhand = 2 if config.get('bimanual', False) else 1

        cameras = config['cameras']
        self.cam_names = list(cameras.keys())
        robot = config['robot']

        # State buffers
        self._rgb = {cam: None for cam in self.cam_names}
        self._depth = {cam: None for cam in self.cam_names}
        self._intrinsics = {cam: None for cam in self.cam_names}
        self._eef_pose = None
        self._gripper_state = 0.0
        self._eef_history = collections.deque(maxlen=self.num_history)

        # Extrinsics (loaded from config or TF)
        self._extrinsics = {cam: np.eye(4, dtype=np.float32) for cam in self.cam_names}

        # TF listener for extrinsics
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.world_frame = robot['world_frame']
        self._cam_tf_frames = {cam: cameras[cam]['tf_frame'] for cam in self.cam_names}

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to cameras
        for cam_name in self.cam_names:
            cam_cfg = cameras[cam_name]
            self.create_subscription(
                Image, cam_cfg['rgb_topic'],
                lambda msg, c=cam_name: self._rgb_callback(msg, c),
                sensor_qos
            )
            self.create_subscription(
                Image, cam_cfg['depth_topic'],
                lambda msg, c=cam_name: self._depth_callback(msg, c),
                sensor_qos
            )
            self.create_subscription(
                CameraInfo, cam_cfg['camera_info_topic'],
                lambda msg, c=cam_name: self._info_callback(msg, c),
                sensor_qos
            )

        # Subscribe to robot state
        self.create_subscription(
            PoseStamped, robot['eef_topic'],
            self._eef_callback, sensor_qos
        )
        self.create_subscription(
            Float64, robot['gripper_topic'],
            self._gripper_callback, sensor_qos
        )

        # Publisher for target pose
        self.target_pub = self.create_publisher(
            PoseStamped, robot['target_pose_topic'], 10
        )

        # Instruction subscriber (can change instruction at runtime)
        self.create_subscription(
            String, '/policy/instruction',
            self._instruction_callback, 10
        )

        # Tokenize instruction
        self._tokenized_instr = None
        self._tokenize_instruction()

        # Inference timer
        rate = config.get('publish_rate', 10.0)
        self.create_timer(1.0 / rate, self._inference_loop)
        self.get_logger().info(
            f"Policy inference node started at {rate} Hz "
            f"with instruction: '{self.instruction}'"
        )

    def _tokenize_instruction(self):
        """Tokenize the current instruction for model input."""
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tokenizer(
            self.instruction, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        self._tokenized_instr = tokens["input_ids"].to(self.device)

    def _rgb_callback(self, msg, cam_name):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.lock:
            self._rgb[cam_name] = img

    def _depth_callback(self, msg, cam_name):
        if msg.encoding == '16UC1':
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        else:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        with self.lock:
            self._depth[cam_name] = depth

    def _info_callback(self, msg, cam_name):
        K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        with self.lock:
            self._intrinsics[cam_name] = K

    def _eef_callback(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        state = np.array(
            [p.x, p.y, p.z, q.x, q.y, q.z, q.w, self._gripper_state],
            dtype=np.float32
        )
        with self.lock:
            self._eef_pose = state
            self._eef_history.append(state.copy())

    def _gripper_callback(self, msg):
        val = float(msg.data)
        if val > 1.0:
            val = val / 255.0
        with self.lock:
            self._gripper_state = val

    def _instruction_callback(self, msg):
        new_instr = msg.data.strip()
        if new_instr and new_instr != self.instruction:
            self.instruction = new_instr
            self._tokenize_instruction()
            self.get_logger().info(f"Instruction updated: '{self.instruction}'")

    def _update_extrinsics(self):
        """Update camera extrinsics from TF tree."""
        for cam_name in self.cam_names:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.world_frame,
                    self._cam_tf_frames[cam_name],
                    rclpy.time.Time()
                )
                t = tf.transform.translation
                r = tf.transform.rotation
                from scipy.spatial.transform import Rotation as R
                rot = R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
                mat = np.eye(4, dtype=np.float32)
                mat[:3, :3] = rot
                mat[:3, 3] = [t.x, t.y, t.z]
                self._extrinsics[cam_name] = mat
            except Exception:
                pass  # keep previous extrinsics

    def _is_ready(self):
        """Check if all sensor data is available."""
        for cam in self.cam_names:
            if self._rgb[cam] is None or self._depth[cam] is None:
                return False
            if self._intrinsics[cam] is None:
                return False
        if self._eef_pose is None:
            return False
        if len(self._eef_history) < self.num_history:
            return False
        return True

    @torch.no_grad()
    def _inference_loop(self):
        """Main inference callback — runs the policy and publishes target pose."""
        with self.lock:
            if not self._is_ready():
                return
            # Snapshot current sensor state
            rgbs = {c: self._rgb[c].copy() for c in self.cam_names}
            depths = {c: self._depth[c].copy() for c in self.cam_names}
            intrinsics = {c: self._intrinsics[c].copy() for c in self.cam_names}
            history = list(self._eef_history)

        self._update_extrinsics()

        # Build model inputs
        ncam = len(self.cam_names)

        # RGB: (1, ncam, 3, H, W)
        rgb_tensors = []
        for cam in self.cam_names:
            rgb_tensors.append(preprocess_rgb(rgbs[cam], self.im_size))
        rgb3d = torch.stack(rgb_tensors).unsqueeze(0).to(self.device)

        # Point clouds: (1, ncam, 3, H, W)
        pcd_tensors = []
        for cam in self.cam_names:
            d = preprocess_depth(depths[cam], self.im_size)
            K = torch.from_numpy(intrinsics[cam]).float()
            E = torch.from_numpy(self._extrinsics[cam]).float()
            pcd = depth_to_pointcloud(d, K, E)
            pcd_tensors.append(pcd)
        pcd = torch.stack(pcd_tensors).unsqueeze(0).to(self.device)

        # Proprioception: (1, nhist, nhand, 8)
        # Pad history if needed
        while len(history) < self.num_history:
            history.insert(0, history[0])
        prop = np.stack(history[-self.num_history:])  # (nhist, 8)
        prop = prop.reshape(self.num_history, self.nhand, 8)
        proprio = torch.from_numpy(prop).float().unsqueeze(0).to(self.device)

        # Trajectory mask: (1, 1, nhand) — all zeros to generate full trajectory
        traj_mask = torch.zeros(1, 1, self.nhand, dtype=torch.bool, device=self.device)

        # Run inference
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            trajectory = self.model(
                None, traj_mask,
                rgb3d, None, pcd,
                self._tokenized_instr, proprio,
                run_inference=True
            )

        # trajectory: (1, T, nhand, 3+4+1) = (1, 1, 1, 8) for keypose_only
        pred = trajectory[0, 0, 0].cpu().numpy()  # (8,): x,y,z, qx,qy,qz,qw, gripper

        # Publish target pose
        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = self.world_frame
        target.pose.position.x = float(pred[0])
        target.pose.position.y = float(pred[1])
        target.pose.position.z = float(pred[2])
        target.pose.orientation.x = float(pred[3])
        target.pose.orientation.y = float(pred[4])
        target.pose.orientation.z = float(pred[5])
        target.pose.orientation.w = float(pred[6])
        self.target_pub.publish(target)


def main():
    args = parse_arguments()
    config = load_yaml_config(args.config)

    instruction = args.instruction or config.get('default_instruction', 'do the task')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(config, device)

    if not HAS_ROS2:
        print("ROS2 not available. Model loaded successfully but cannot run inference node.")
        print("Install rclpy and sensor_msgs to use the ROS2 inference node.")
        return

    # Start ROS2 node
    rclpy.init()
    node = PolicyInferenceNode(model, config, device, instruction)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
