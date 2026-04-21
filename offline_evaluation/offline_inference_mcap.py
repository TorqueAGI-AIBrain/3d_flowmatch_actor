# offline_inference_mcap.py: Read source MCAP bags, run 3DFA inference, write output MCAP.
# offline_inference_mcap.py: Produces per-camera point clouds and GT/predicted trajectories.

"""
Offline inference on raw ROS2 MCAP bags. For each validation episode:
  1. Read synced RGB, depth, EEF, gripper from the source bag
  2. Preprocess (resize, depth→metres, build point clouds)
  3. Run 3DFA model to predict EEF keyposes
  4. Write output MCAP with point clouds and trajectory overlays

Output topics per episode:
    /front/points       - sensor_msgs/PointCloud2 (world frame)
    /wrist/points       - sensor_msgs/PointCloud2 (world frame)
    /gt/trajectory      - geometry_msgs/PoseArray  (accumulated GT)
    /pred/trajectory    - geometry_msgs/PoseArray  (accumulated predicted)

Usage:
    python -m evaluation.offline_inference_mcap --config configs/evaluation.yaml
"""

import argparse
import os
from collections import deque
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mcap.reader import make_reader
from mcap.writer import Writer as McapWriter
from rosbags.typesys import Stores, get_typestore
from modeling.policy import fetch_model_class
from modeling.encoder.text import fetch_tokenizers
from utils.config import load_yaml_config, flatten_config
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud


IM_SIZE = 256
TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
serialize_cdr = TYPESTORE.serialize_cdr
deserialize_cdr = TYPESTORE.deserialize_cdr


# ---------------------------------------------------------------------------
# MCAP reading helpers (bypass rosbags AnyReader which chokes on rosbag2 v9)
# ---------------------------------------------------------------------------

def _read_mcap_messages(mcap_path, topic_filter=None):
    """Read all messages from an MCAP file, return {topic: [(time_sec, decoded_msg)]}."""
    result = {}
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if topic_filter and channel.topic not in topic_filter:
                continue
            topic = channel.topic
            if topic not in result:
                result[topic] = []
            msg = deserialize_cdr(message.data, schema.name)
            t_sec = message.log_time / 1e9
            result[topic].append((t_sec, msg))
    return result


def _find_closest_msg(target_time, messages, slop):
    """Find the message closest to target_time within slop tolerance."""
    best = None
    best_diff = float('inf')
    for msg_time, msg in messages:
        diff = abs(msg_time - target_time)
        if diff < best_diff and diff <= slop:
            best_diff = diff
            best = msg
    return best


def _decode_image(msg):
    """Decode sensor_msgs/Image to numpy array."""
    encoding = msg.encoding
    h, w = msg.height, msg.width
    if encoding in ('rgb8', 'RGB8'):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
    elif encoding in ('bgr8', 'BGR8'):
        return cv2.cvtColor(
            np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3),
            cv2.COLOR_BGR2RGB
        )
    elif encoding in ('16UC1',):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    elif encoding in ('32FC1',):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
    elif encoding in ('bgra8', 'BGRA8'):
        return cv2.cvtColor(
            np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4),
            cv2.COLOR_BGRA2RGB
        )
    elif encoding in ('rgba8', 'RGBA8'):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")


def _decode_pose(msg):
    """Extract [x,y,z,qx,qy,qz,qw] from PoseStamped."""
    p = msg.pose.position
    q = msg.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)


def _decode_gripper(msg):
    """Extract gripper state as float [0, 1]."""
    val = float(msg.data)
    if val > 1.0:
        val = val / 255.0
    return np.float32(val)


def _decode_camera_info(msg):
    """Extract 3x3 intrinsic matrix from CameraInfo."""
    return np.array(msg.k, dtype=np.float32).reshape(3, 3)


# ---------------------------------------------------------------------------
# Args & config
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline MCAP inference with 3DFA model"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--bag_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, nargs="+", default=None)
    parser.add_argument("--instruction", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    """Load YAML config and apply CLI overrides."""
    cfg = load_yaml_config(args.config)
    flat = flatten_config(cfg)
    # CLI overrides
    if args.checkpoint:
        flat['checkpoint'] = args.checkpoint
    if args.bag_dir:
        flat['bag_dir'] = args.bag_dir
    if args.output_dir:
        flat['output_dir'] = args.output_dir
    if args.instruction:
        flat['instruction'] = args.instruction
    # Keep nested topics dict
    flat['topics'] = cfg.get('topics', {})
    # Keep val_episodes as list
    flat['val_episodes'] = args.episodes or cfg.get('val_episodes', [9, 20, 32])
    return flat


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg, device):
    """Build and load 3DFA model from checkpoint."""
    model_cls = fetch_model_class(cfg['model_type'])
    model = model_cls(
        backbone=cfg['backbone'],
        finetune_backbone=cfg.get('finetune_backbone', False),
        finetune_text_encoder=cfg.get('finetune_text_encoder', False),
        num_vis_instr_attn_layers=cfg['num_vis_instr_attn_layers'],
        fps_subsampling_factor=cfg['fps_subsampling_factor'],
        embedding_dim=cfg['embedding_dim'],
        num_attn_heads=cfg['num_attn_heads'],
        nhist=cfg['num_history'],
        nhand=2 if cfg.get('bimanual', False) else 1,
        num_shared_attn_layers=cfg['num_shared_attn_layers'],
        relative=cfg.get('relative_action', False),
        rotation_format=cfg['rotation_format'],
        denoise_timesteps=cfg['denoise_timesteps'],
        denoise_model=cfg['denoise_model'],
    )
    ckpt = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    weights = {k.replace('module.', ''): v for k, v in ckpt['weight'].items()}
    model.load_state_dict(weights, strict=False)
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {cfg['checkpoint']} (iter {ckpt.get('iter', '?')})")
    return model


# ---------------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------------

def pose_7d_to_4x4(pose):
    """Convert [x,y,z,qx,qy,qz,qw] to 4x4 homogeneous matrix."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def scale_intrinsics(K, orig_w, orig_h, target_size):
    """Scale intrinsic matrix from original resolution to target_size x target_size."""
    K = K.copy()
    K[0, 0] *= target_size / orig_w   # fx
    K[1, 1] *= target_size / orig_h   # fy
    K[0, 2] *= target_size / orig_w   # cx
    K[1, 2] *= target_size / orig_h   # cy
    return K


def read_episode_bag(bag_dir, topics, target_hz, sync_slop):
    """Read and synchronize all frames from one episode MCAP bag.

    Returns dict with per-frame lists, or None if the bag is invalid.
    """
    bag_dir = Path(bag_dir)
    # Find the .mcap file inside the bag directory
    mcap_files = list(bag_dir.glob('*.mcap'))
    if not mcap_files:
        print(f"  No .mcap file found in {bag_dir}")
        return None
    mcap_path = mcap_files[0]

    all_topics = set(topics.values())
    msgs = _read_mcap_messages(str(mcap_path), topic_filter=all_topics)

    eef_msgs = msgs[topics['eef_pose']]
    if len(eef_msgs) < 2:
        print(f"  Skipping {bag_path}: too few EEF messages ({len(eef_msgs)})")
        return None

    # Time reference from EEF, resample at target_hz
    t_start = eef_msgs[0][0]
    t_end = eef_msgs[-1][0]
    sample_times = np.arange(t_start, t_end, 1.0 / target_hz)
    if len(sample_times) < 2:
        return None

    # Static: front extrinsic (first /front/pose), intrinsics (first CameraInfo)
    front_pose_msgs = msgs[topics['front_pose']]
    front_E = pose_7d_to_4x4(_decode_pose(front_pose_msgs[0][1]))

    # Scale intrinsics from original resolution to IM_SIZE
    front_info_msgs = msgs[topics['front_info']]
    wrist_info_msgs = msgs[topics['wrist_info']]
    first_front_rgb_msg = msgs[topics['front_rgb']][0][1]
    first_wrist_rgb_msg = msgs[topics['wrist_rgb']][0][1]
    front_K = scale_intrinsics(
        _decode_camera_info(front_info_msgs[0][1]),
        first_front_rgb_msg.width, first_front_rgb_msg.height, IM_SIZE
    )
    wrist_K = scale_intrinsics(
        _decode_camera_info(wrist_info_msgs[0][1]),
        first_wrist_rgb_msg.width, first_wrist_rgb_msg.height, IM_SIZE
    )

    # Synchronize frames
    frames = []
    gripper_msgs = msgs[topics['gripper']]

    for t_sample in sample_times:
        eef_msg = _find_closest_msg(t_sample, eef_msgs, sync_slop * 2)
        if eef_msg is None:
            continue

        front_rgb = _find_closest_msg(t_sample, msgs[topics['front_rgb']], sync_slop)
        front_depth = _find_closest_msg(t_sample, msgs[topics['front_depth']], sync_slop)
        wrist_rgb = _find_closest_msg(t_sample, msgs[topics['wrist_rgb']], sync_slop)
        wrist_depth = _find_closest_msg(t_sample, msgs[topics['wrist_depth']], sync_slop)
        wrist_pose = _find_closest_msg(t_sample, msgs[topics['wrist_pose']], sync_slop)
        grip_msg = _find_closest_msg(t_sample, gripper_msgs, sync_slop * 2)

        if any(m is None for m in [front_rgb, front_depth, wrist_rgb, wrist_depth, wrist_pose]):
            continue

        eef_pose = _decode_pose(eef_msg)  # (7,)
        gripper = _decode_gripper(grip_msg) if grip_msg else np.float32(0.0)

        frames.append({
            'front_rgb': _decode_image(front_rgb),
            'front_depth': _decode_image(front_depth),
            'wrist_rgb': _decode_image(wrist_rgb),
            'wrist_depth': _decode_image(wrist_depth),
            'wrist_E': pose_7d_to_4x4(_decode_pose(wrist_pose)),
            'eef_pose': np.concatenate([eef_pose, [gripper]]),  # (8,)
            'timestamp': t_sample - t_start,  # relative seconds
        })

    if len(frames) < 2:
        print(f"  Skipping {bag_path}: too few synced frames ({len(frames)})")
        return None

    return {
        'frames': frames,
        'front_E': front_E,
        'front_K': front_K,
        'wrist_K': wrist_K,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def resize_rgb(img, size):
    """Resize (H,W,3) uint8 RGB to (size, size)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def resize_depth(depth, size):
    """Resize depth map to (size, size) with nearest neighbour."""
    return cv2.resize(depth.astype(np.float32), (size, size),
                      interpolation=cv2.INTER_NEAREST)


def preprocess_frame(frame, front_E, front_K, wrist_K, eef_history,
                     depth2cloud, depth_scale, device):
    """Convert one frame of raw data into model-ready tensors.

    Returns (rgbs, pcds, proprio) tensors on device.
    """
    # RGB: resize, (3,H,W), [0,1]
    f_rgb = resize_rgb(frame['front_rgb'], IM_SIZE)
    w_rgb = resize_rgb(frame['wrist_rgb'], IM_SIZE)
    rgbs = np.stack([
        f_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0,
        w_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0,
    ])  # (2, 3, 256, 256)
    rgbs = torch.from_numpy(rgbs).unsqueeze(0).to(device)  # (1, 2, 3, 256, 256)

    # Depth: resize, convert to metres
    f_depth = resize_depth(frame['front_depth'], IM_SIZE) / depth_scale
    w_depth = resize_depth(frame['wrist_depth'], IM_SIZE) / depth_scale
    depth = np.stack([f_depth, w_depth])  # (2, 256, 256)
    depth = torch.from_numpy(depth).unsqueeze(0).to(device)  # (1, 2, 256, 256)

    # Extrinsics and intrinsics
    extr = np.stack([front_E, frame['wrist_E']])  # (2, 4, 4)
    intr = np.stack([front_K, wrist_K])  # (2, 3, 3)
    extr = torch.from_numpy(extr).unsqueeze(0).to(device)  # (1, 2, 4, 4)
    intr = torch.from_numpy(intr).unsqueeze(0).to(device)  # (1, 2, 3, 3)

    # Point clouds
    pcds = depth2cloud(depth, extr, intr)  # (1, 2, 3, 256, 256)

    # Proprio: last N history states → (1, N, 1, 8)
    proprio_np = np.stack(list(eef_history))  # (N, 8)
    proprio = torch.from_numpy(proprio_np).unsqueeze(0).unsqueeze(2).to(device)
    # (1, num_history, 1, 8)

    return rgbs, pcds, proprio, f_rgb, w_rgb


# ---------------------------------------------------------------------------
# MCAP writing helpers
# ---------------------------------------------------------------------------

def build_pointcloud2(points, colors, stamp, frame_id, typestore=TYPESTORE):
    """Build sensor_msgs/PointCloud2 from (N,3) xyz and (N,3) uint8 rgb."""
    PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
    PointField = typestore.types['sensor_msgs/msg/PointField']
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']

    # Filter invalid points
    valid = np.isfinite(points).all(axis=1) & (np.abs(points) < 10.0).all(axis=1)
    points = points[valid]
    colors = colors[valid]

    N = len(points)
    if N == 0:
        N = 1
        points = np.zeros((1, 3), dtype=np.float32)
        colors = np.zeros((1, 3), dtype=np.uint8)

    # Pack RGB into float32: (r<<16 | g<<8 | b) viewed as float
    rgb_packed = (colors[:, 0].astype(np.uint32) << 16 |
                  colors[:, 1].astype(np.uint32) << 8 |
                  colors[:, 2].astype(np.uint32))
    rgb_float = rgb_packed.view(np.float32)

    # Build data buffer: x,y,z (float32) + rgb (float32) = 16 bytes per point
    data = np.empty(N, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    data['rgb'] = rgb_float

    fields = [
        PointField(name='x', offset=0, datatype=7, count=1),  # FLOAT32
        PointField(name='y', offset=4, datatype=7, count=1),
        PointField(name='z', offset=8, datatype=7, count=1),
        PointField(name='rgb', offset=12, datatype=7, count=1),
    ]

    header = Header(
        stamp=Time(sec=int(stamp), nanosec=int((stamp % 1) * 1e9)),
        frame_id=frame_id,
    )

    return PointCloud2(
        header=header,
        height=1,
        width=N,
        fields=fields,
        is_bigendian=False,
        point_step=16,
        row_step=N * 16,
        data=np.frombuffer(data.tobytes(), dtype=np.uint8),
        is_dense=True,
    )


def build_pose_array(poses_7d, stamp, frame_id, typestore=TYPESTORE):
    """Build geometry_msgs/PoseArray from list of (7,) [x,y,z,qx,qy,qz,qw]."""
    PoseArray = typestore.types['geometry_msgs/msg/PoseArray']
    Pose = typestore.types['geometry_msgs/msg/Pose']
    Point = typestore.types['geometry_msgs/msg/Point']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']

    header = Header(
        stamp=Time(sec=int(stamp), nanosec=int((stamp % 1) * 1e9)),
        frame_id=frame_id,
    )

    poses = []
    for p in poses_7d:
        poses.append(Pose(
            position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
            orientation=Quaternion(
                x=float(p[3]), y=float(p[4]), z=float(p[5]), w=float(p[6])
            ),
        ))

    return PoseArray(header=header, poses=poses)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectories(gt_arr, pred_arr, pos_error, ep_idx, output_dir):
    """Plot GT vs predicted EEF waypoints in 3D and save to PNG."""
    gt_pos = gt_arr[:, :3]
    pred_pos = pred_arr[:, :3]
    N = len(gt_pos)
    t = np.linspace(0, 1, N)

    fig = plt.figure(figsize=(16, 6))

    # Color by time progression
    cmap_gt = plt.cm.Greens(np.linspace(0.3, 1.0, N))
    cmap_pred = plt.cm.Reds(np.linspace(0.3, 1.0, N))

    # --- 3D waypoints ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
                c=cmap_gt, s=15, alpha=0.8, label='GT', depthshade=False)
    ax1.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2],
                c=cmap_pred, s=15, alpha=0.8, label='Pred', depthshade=False)
    # Start/end markers
    ax1.scatter(*gt_pos[0], color='#2ecc71', s=80, marker='o', edgecolors='k', zorder=5)
    ax1.scatter(*gt_pos[-1], color='#2ecc71', s=80, marker='D', edgecolors='k', zorder=5)
    ax1.scatter(*pred_pos[0], color='#e74c3c', s=80, marker='o', edgecolors='k', zorder=5)
    ax1.scatter(*pred_pos[-1], color='#e74c3c', s=80, marker='D', edgecolors='k', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Episode {ep_idx} — 3D Waypoints')
    ax1.legend(fontsize=9)
    ax1.view_init(elev=25, azim=-60)

    # --- XY projection ---
    ax2 = fig.add_subplot(132)
    ax2.scatter(gt_pos[:, 0], gt_pos[:, 1], c=cmap_gt, s=15, alpha=0.8, label='GT')
    ax2.scatter(pred_pos[:, 0], pred_pos[:, 1], c=cmap_pred, s=15, alpha=0.8, label='Pred')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection')
    ax2.legend(fontsize=9)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # --- Error over time ---
    ax3 = fig.add_subplot(133)
    ax3.fill_between(range(N), 0, pos_error * 100, alpha=0.3, color='#3498db')
    ax3.plot(pos_error * 100, '-', color='#3498db', linewidth=1.5)
    ax3.axhline(y=pos_error.mean() * 100, color='#e74c3c', linestyle='--',
                linewidth=1, label=f'Mean: {pos_error.mean()*100:.2f}cm')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Error (cm)')
    ax3.set_title('Position Error')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f'Episode {ep_idx} | {N} frames | '
                 f'Mean err: {pos_error.mean()*100:.2f}cm | '
                 f'Max err: {pos_error.max()*100:.2f}cm',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'episode_{ep_idx}_trajectory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

def process_episode(ep_idx, cfg, model, tokenizer, depth2cloud, device):
    """Run inference on one episode and write output MCAP."""
    bag_dir = Path(cfg['bag_dir']) / f"episode_{ep_idx}_bag"
    if not bag_dir.exists():
        print(f"  Bag not found: {bag_dir}")
        return

    print(f"\n=== Episode {ep_idx} ===")

    # Read bag
    episode = read_episode_bag(
        bag_dir, cfg['topics'],
        cfg.get('target_hz', 10.0),
        cfg.get('sync_slop', 0.05),
    )
    if episode is None:
        return

    frames = episode['frames']
    num_history = cfg['num_history']
    depth_scale = cfg.get('depth_scale', 1000.0)
    print(f"  {len(frames)} synced frames")

    # Tokenize instruction
    instr_tokens = tokenizer([cfg.get('instruction', 'do the task')]).to(device)

    # Proprio history buffer
    eef_history = deque(maxlen=num_history)

    # Trajectory accumulators
    gt_poses = []
    pred_poses = []

    # Output MCAP — single .mcap file per episode
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    mcap_path = output_dir / f"episode_{ep_idx}.mcap"

    mcap_file = open(mcap_path, 'wb')
    writer = McapWriter(mcap_file)
    writer.start(profile='ros2', library='offline_inference_mcap')

    # Register schemas and channels with full ROS2 message definitions.
    # Foxglove requires the concatenated .msg text for CDR-encoded messages.
    _MSG_DEFS = {
        'sensor_msgs/msg/PointCloud2': (
            "# sensor_msgs/msg/PointCloud2\n"
            "std_msgs/Header header\n"
            "uint32 height\n"
            "uint32 width\n"
            "sensor_msgs/PointField[] fields\n"
            "bool is_bigendian\n"
            "uint32 point_step\n"
            "uint32 row_step\n"
            "uint8[] data\n"
            "bool is_dense\n"
            "\n"
            "================================================================================\n"
            "MSG: std_msgs/Header\n"
            "builtin_interfaces/Time stamp\n"
            "string frame_id\n"
            "\n"
            "================================================================================\n"
            "MSG: builtin_interfaces/Time\n"
            "int32 sec\n"
            "uint32 nanosec\n"
            "\n"
            "================================================================================\n"
            "MSG: sensor_msgs/PointField\n"
            "string name\n"
            "uint32 offset\n"
            "uint8 datatype\n"
            "uint32 count\n"
        ),
        'geometry_msgs/msg/PoseArray': (
            "# geometry_msgs/msg/PoseArray\n"
            "std_msgs/Header header\n"
            "geometry_msgs/Pose[] poses\n"
            "\n"
            "================================================================================\n"
            "MSG: std_msgs/Header\n"
            "builtin_interfaces/Time stamp\n"
            "string frame_id\n"
            "\n"
            "================================================================================\n"
            "MSG: builtin_interfaces/Time\n"
            "int32 sec\n"
            "uint32 nanosec\n"
            "\n"
            "================================================================================\n"
            "MSG: geometry_msgs/Pose\n"
            "geometry_msgs/Point position\n"
            "geometry_msgs/Quaternion orientation\n"
            "\n"
            "================================================================================\n"
            "MSG: geometry_msgs/Point\n"
            "float64 x\n"
            "float64 y\n"
            "float64 z\n"
            "\n"
            "================================================================================\n"
            "MSG: geometry_msgs/Quaternion\n"
            "float64 x\n"
            "float64 y\n"
            "float64 z\n"
            "float64 w\n"
        ),
    }

    def _register_topic(topic, msgtype):
        schema_id = writer.register_schema(
            name=msgtype, encoding='ros2msg',
            data=_MSG_DEFS[msgtype].encode('utf-8'),
        )
        return writer.register_channel(
            topic=topic, message_encoding='cdr',
            schema_id=schema_id,
        )

    front_pc_ch = _register_topic('/front/points', 'sensor_msgs/msg/PointCloud2')
    wrist_pc_ch = _register_topic('/wrist/points', 'sensor_msgs/msg/PointCloud2')
    gt_traj_ch = _register_topic('/gt/trajectory', 'geometry_msgs/msg/PoseArray')
    pred_traj_ch = _register_topic('/pred/trajectory', 'geometry_msgs/msg/PoseArray')

    try:

        for i, frame in enumerate(tqdm(frames, desc=f"  Inference")):
            # Update proprio history:
            # First frame seeds from GT, subsequent frames use previous prediction
            eef_state_gt = frame['eef_pose']  # (8,) GT for reference
            if len(eef_history) == 0:
                # Seed with GT for the first frame (no prediction yet)
                for _ in range(num_history):
                    eef_history.append(eef_state_gt.copy())
            # else: already updated after previous prediction (see below)

            # Preprocess (RGB/depth always from bag, proprio from autoregressive buffer)
            rgbs, pcds, proprio, f_rgb_np, w_rgb_np = preprocess_frame(
                frame, episode['front_E'], episode['front_K'], episode['wrist_K'],
                eef_history, depth2cloud, depth_scale, device,
            )

            # Model inference
            action_mask = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(
                    None, action_mask, rgbs, None, pcds,
                    instr_tokens, proprio[:, :, :, :7],
                    run_inference=True,
                )
            pred_np = pred[0, 0, 0].cpu().float().numpy()  # (8,)

            # Feed prediction back into proprio history (autoregressive)
            eef_history.append(pred_np.copy())

            # Accumulate trajectories
            gt_poses.append(eef_state_gt[:7].copy())
            pred_poses.append(pred_np[:7].copy())

            # Build point clouds from pcds tensor
            # pcds shape: (1, 2, 3, 256, 256) — [front, wrist]
            pcds_np = pcds[0].cpu().float().numpy()  # (2, 3, 256, 256)

            front_pts = pcds_np[0].reshape(3, -1).T   # (N, 3)
            wrist_pts = pcds_np[1].reshape(3, -1).T

            front_colors = f_rgb_np.reshape(-1, 3)     # (N, 3) uint8
            wrist_colors = w_rgb_np.reshape(-1, 3)

            # Timestamp
            t = frame['timestamp']
            t_ns = int(t * 1e9)

            # Build and write messages
            front_pc_msg = build_pointcloud2(front_pts, front_colors, t, 'base_link')
            wrist_pc_msg = build_pointcloud2(wrist_pts, wrist_colors, t, 'base_link')
            gt_traj_msg = build_pose_array(gt_poses, t, 'base_link')
            pred_traj_msg = build_pose_array(pred_poses, t, 'base_link')

            writer.add_message(front_pc_ch, t_ns, serialize_cdr(front_pc_msg, front_pc_msg.__msgtype__), t_ns)
            writer.add_message(wrist_pc_ch, t_ns, serialize_cdr(wrist_pc_msg, wrist_pc_msg.__msgtype__), t_ns)
            writer.add_message(gt_traj_ch, t_ns, serialize_cdr(gt_traj_msg, gt_traj_msg.__msgtype__), t_ns)
            writer.add_message(pred_traj_ch, t_ns, serialize_cdr(pred_traj_msg, pred_traj_msg.__msgtype__), t_ns)

    finally:
        writer.finish()
        mcap_file.close()

    print(f"  Saved {len(frames)} frames to {mcap_path}")

    # Print summary metrics and plot trajectories
    gt_arr = np.array(gt_poses)
    pred_arr = np.array(pred_poses)
    pos_error = np.linalg.norm(gt_arr[:, :3] - pred_arr[:, :3], axis=1)
    print(f"  Position error: mean={pos_error.mean()*100:.2f}cm, "
          f"max={pos_error.max()*100:.2f}cm")

    plot_trajectories(gt_arr, pred_arr, pos_error, ep_idx, output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading model...")
    model = load_model(cfg, device)
    tokenizer = fetch_tokenizers(cfg['backbone'])
    depth2cloud = RLBenchDepth2Cloud((IM_SIZE, IM_SIZE))

    os.makedirs(cfg['output_dir'], exist_ok=True)

    for ep_idx in cfg['val_episodes']:
        process_episode(ep_idx, cfg, model, tokenizer, depth2cloud, device)

    print(f"\nDone. Output at {cfg['output_dir']}")


if __name__ == '__main__':
    main()
