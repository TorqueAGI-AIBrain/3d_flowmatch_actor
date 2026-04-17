#!/usr/bin/env python3
"""
ROS2 node for recording real robot arm demonstrations.

Outputs the exact directory structure expected by XArmDataset:

    <output_dir>/<task_name>/
        episode_N/
            rgb/<cam>_0000.png
            depth/<cam>_0000.png
            eef_states.npy          # (T, 8) [x,y,z, qx,qy,qz,qw, gripper_open]
            camera_extrinsics.npy   # (ncam, 4, 4)
            camera_intrinsics.npy   # (ncam, 3, 3)
        instructions.json

=== Topic Configuration ===

Configure via ROS2 parameters or the defaults below.
Typical RealSense + robot arm topics:

    RGB:          /camera_front/color/image_raw        (sensor_msgs/Image)
    Depth:        /camera_front/aligned_depth_to_color/image_raw  (sensor_msgs/Image, 16UC1)
    CameraInfo:   /camera_front/color/camera_info      (sensor_msgs/CameraInfo)
    EEF Pose:     /end_effector_pose                   (geometry_msgs/PoseStamped)
    Gripper:      /gripper/state                       (std_msgs/Float64 or Bool)
    TF:           /tf, /tf_static                      (for extrinsics via TF tree)

=== Usage ===

    ros2 run data_processing ros2_record_demos \
        --ros-args \
        -p task_name:=pick_cup \
        -p output_dir:=/path/to/demos \
        -p cameras:="['front', 'wrist']" \
        -p hz:=10.0

    Keyboard controls during recording:
        ENTER  = start/stop episode
        q      = quit and save

    Or trigger via ROS2 services:
        ros2 service call /recorder/start_episode std_srvs/srv/Trigger
        ros2 service call /recorder/stop_episode  std_srvs/srv/Trigger
"""

import os
import json
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from std_srvs.srv import Trigger

from cv_bridge import CvBridge
import cv2

# If using TF for extrinsics
from tf2_ros import Buffer, TransformListener


IM_SIZE = 256


class DemoRecorder(Node):

    def __init__(self):
        super().__init__("demo_recorder")

        # ----- Parameters -----
        self.declare_parameter("task_name", "default_task")
        self.declare_parameter("output_dir", "/tmp/robot_demos")
        self.declare_parameter("cameras", ["front", "wrist"])
        self.declare_parameter("hz", 10.0)
        self.declare_parameter("instruction", "do the task")
        self.declare_parameter("variation", 0)
        self.declare_parameter("world_frame", "base_link")
        self.declare_parameter("eef_frame", "tool0")
        self.declare_parameter("nhand", 1)

        # --- Topic overrides (set per camera via <cam>_rgb_topic, etc.) ---
        # Defaults assume RealSense naming: /camera_<cam>/...
        self.declare_parameter("eef_topic", "/end_effector_pose")
        self.declare_parameter("gripper_topic", "/gripper/state")

        self.task_name = self.get_parameter("task_name").value
        self.output_dir = self.get_parameter("output_dir").value
        self.cameras = self.get_parameter("cameras").value
        self.hz = self.get_parameter("hz").value
        self.instruction = self.get_parameter("instruction").value
        self.variation = self.get_parameter("variation").value
        self.world_frame = self.get_parameter("world_frame").value
        self.eef_frame = self.get_parameter("eef_frame").value
        self.nhand = self.get_parameter("nhand").value

        self.bridge = CvBridge()
        self.lock = Lock()

        # ----- State -----
        self.recording = False
        self.episode_count = self._count_existing_episodes()
        self.current_episode = {
            "rgb": {cam: [] for cam in self.cameras},
            "depth": {cam: [] for cam in self.cameras},
            "eef_states": [],
        }
        self.camera_intrinsics = {}   # cam_name -> (3, 3)
        self.camera_extrinsics = {}   # cam_name -> (4, 4)
        self.latest_eef_pose = None   # (7,) [x,y,z, qx,qy,qz,qw]
        self.latest_gripper = 1.0     # 0=closed, 1=open
        self.latest_rgb = {}          # cam -> np.ndarray
        self.latest_depth = {}        # cam -> np.ndarray

        # ----- QoS -----
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ----- Subscribers per camera -----
        for cam in self.cameras:
            # RGB
            rgb_topic = f"/camera_{cam}/color/image_raw"
            self.declare_parameter(f"{cam}_rgb_topic", rgb_topic)
            rgb_topic = self.get_parameter(f"{cam}_rgb_topic").value
            self.create_subscription(
                Image, rgb_topic,
                lambda msg, c=cam: self._rgb_cb(msg, c),
                sensor_qos,
            )

            # Depth
            depth_topic = f"/camera_{cam}/aligned_depth_to_color/image_raw"
            self.declare_parameter(f"{cam}_depth_topic", depth_topic)
            depth_topic = self.get_parameter(f"{cam}_depth_topic").value
            self.create_subscription(
                Image, depth_topic,
                lambda msg, c=cam: self._depth_cb(msg, c),
                sensor_qos,
            )

            # CameraInfo (for intrinsics)
            info_topic = f"/camera_{cam}/color/camera_info"
            self.declare_parameter(f"{cam}_info_topic", info_topic)
            info_topic = self.get_parameter(f"{cam}_info_topic").value
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, c=cam: self._camera_info_cb(msg, c),
                sensor_qos,
            )

        # ----- EEF pose subscriber -----
        eef_topic = self.get_parameter("eef_topic").value
        self.create_subscription(
            PoseStamped, eef_topic,
            self._eef_cb, sensor_qos,
        )

        # ----- Gripper state subscriber -----
        gripper_topic = self.get_parameter("gripper_topic").value
        self.create_subscription(
            Float64, gripper_topic,
            self._gripper_cb, sensor_qos,
        )

        # ----- TF for camera extrinsics -----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----- Services for external triggering -----
        self.create_service(Trigger, "~/start_episode", self._srv_start)
        self.create_service(Trigger, "~/stop_episode", self._srv_stop)

        # ----- Recording timer -----
        period = 1.0 / self.hz
        self.timer = self.create_timer(period, self._record_tick)

        # ----- Keyboard input timer (non-blocking) -----
        self.create_timer(0.1, self._keyboard_poll)

        self.get_logger().info(
            f"DemoRecorder ready. Task: {self.task_name}, "
            f"Cameras: {self.cameras}, Hz: {self.hz}"
        )
        self.get_logger().info("Press ENTER to start/stop episode, 'q' to quit.")

    # ===================== Callbacks =====================

    def _rgb_cb(self, msg: Image, cam: str):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.resize(img, (IM_SIZE, IM_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.lock:
            self.latest_rgb[cam] = img

    def _depth_cb(self, msg: Image, cam: str):
        # Expect 16UC1 (millimeters) from RealSense
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = cv2.resize(depth, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_NEAREST)
        with self.lock:
            self.latest_depth[cam] = depth.astype(np.uint16)

    def _camera_info_cb(self, msg: CameraInfo, cam: str):
        K = np.array(msg.k).reshape(3, 3)
        with self.lock:
            self.camera_intrinsics[cam] = K.astype(np.float32)

    def _eef_cb(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        with self.lock:
            # Store as [x, y, z, qx, qy, qz, qw]
            self.latest_eef_pose = np.array([
                p.x, p.y, p.z,
                q.x, q.y, q.z, q.w,
            ], dtype=np.float32)

    def _gripper_cb(self, msg: Float64):
        with self.lock:
            # Normalize: >0.5 = open (1.0), <=0.5 = closed (0.0)
            # Adjust this logic for your specific gripper
            self.latest_gripper = float(msg.data)

    # ===================== TF Extrinsics =====================

    def _lookup_extrinsics(self):
        """Look up camera-to-world transform from TF tree for each camera."""
        for cam in self.cameras:
            if cam in self.camera_extrinsics:
                continue
            cam_frame = f"camera_{cam}_color_optical_frame"
            self.declare_parameter(f"{cam}_tf_frame", cam_frame)
            cam_frame = self.get_parameter(f"{cam}_tf_frame").value
            try:
                t = self.tf_buffer.lookup_transform(
                    self.world_frame, cam_frame, rclpy.time.Time()
                )
                trans = t.transform.translation
                rot = t.transform.rotation
                # Build 4x4 from translation + quaternion
                from scipy.spatial.transform import Rotation as R
                mat = np.eye(4, dtype=np.float32)
                mat[:3, :3] = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
                mat[:3, 3] = [trans.x, trans.y, trans.z]
                self.camera_extrinsics[cam] = mat
                self.get_logger().info(f"Got extrinsics for {cam}")
            except Exception:
                pass  # Will retry on next tick

    # ===================== Recording Logic =====================

    def _record_tick(self):
        """Called at self.hz. If recording, snapshot all data."""
        if not self.recording:
            self._lookup_extrinsics()
            return

        with self.lock:
            # Check all data is available
            if self.latest_eef_pose is None:
                self.get_logger().warn("No EEF pose yet, skipping frame", throttle_duration_sec=2.0)
                return
            for cam in self.cameras:
                if cam not in self.latest_rgb or cam not in self.latest_depth:
                    self.get_logger().warn(
                        f"Missing image for {cam}, skipping frame",
                        throttle_duration_sec=2.0,
                    )
                    return

            # Snapshot
            eef = np.concatenate([
                self.latest_eef_pose,
                np.array([self.latest_gripper], dtype=np.float32),
            ])  # (8,)
            self.current_episode["eef_states"].append(eef.copy())

            for cam in self.cameras:
                self.current_episode["rgb"][cam].append(self.latest_rgb[cam].copy())
                self.current_episode["depth"][cam].append(self.latest_depth[cam].copy())

        t = len(self.current_episode["eef_states"])
        if t % 50 == 0:
            self.get_logger().info(f"  Recording... {t} frames")

    def start_episode(self):
        if self.recording:
            self.get_logger().warn("Already recording!")
            return False

        # Check prerequisites
        missing_intr = [c for c in self.cameras if c not in self.camera_intrinsics]
        missing_extr = [c for c in self.cameras if c not in self.camera_extrinsics]
        if missing_intr:
            self.get_logger().error(
                f"Missing intrinsics for cameras: {missing_intr}. "
                "Check that CameraInfo topics are publishing."
            )
            return False
        if missing_extr:
            self.get_logger().warn(
                f"Missing extrinsics for cameras: {missing_extr}. "
                "Will save identity matrices — provide camera_extrinsics.npy manually."
            )

        self.current_episode = {
            "rgb": {cam: [] for cam in self.cameras},
            "depth": {cam: [] for cam in self.cameras},
            "eef_states": [],
        }
        self.recording = True
        self.get_logger().info(f"=== Episode {self.episode_count} STARTED ===")
        return True

    def stop_episode(self):
        if not self.recording:
            self.get_logger().warn("Not recording!")
            return False

        self.recording = False
        n_frames = len(self.current_episode["eef_states"])
        self.get_logger().info(
            f"=== Episode {self.episode_count} STOPPED ({n_frames} frames) ==="
        )

        if n_frames < 2:
            self.get_logger().warn("Episode too short (<2 frames), discarding.")
            return False

        self._save_episode()
        self.episode_count += 1
        return True

    # ===================== Save to Disk =====================

    def _save_episode(self):
        task_dir = os.path.join(self.output_dir, self.task_name)
        ep_dir = os.path.join(task_dir, f"episode_{self.episode_count}")
        rgb_dir = os.path.join(ep_dir, "rgb")
        depth_dir = os.path.join(ep_dir, "depth")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        n_frames = len(self.current_episode["eef_states"])

        # Save images
        for cam in self.cameras:
            for t in range(n_frames):
                # RGB as PNG
                img = self.current_episode["rgb"][cam][t]
                cv2.imwrite(
                    os.path.join(rgb_dir, f"{cam}_{t:04d}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
                # Depth as 16-bit PNG
                depth = self.current_episode["depth"][cam][t]
                cv2.imwrite(
                    os.path.join(depth_dir, f"{cam}_{t:04d}.png"),
                    depth,
                )

        # EEF states: (T, 8) for single arm, (T, 16) for bimanual
        eef = np.stack(self.current_episode["eef_states"])  # (T, 8)
        np.save(os.path.join(ep_dir, "eef_states.npy"), eef)

        # Camera intrinsics: (ncam, 3, 3)
        intrinsics = np.stack([
            self.camera_intrinsics[cam] for cam in self.cameras
        ])
        np.save(os.path.join(ep_dir, "camera_intrinsics.npy"), intrinsics)

        # Camera extrinsics: (ncam, 4, 4)
        extrinsics = np.stack([
            self.camera_extrinsics.get(cam, np.eye(4, dtype=np.float32))
            for cam in self.cameras
        ])
        np.save(os.path.join(ep_dir, "camera_extrinsics.npy"), extrinsics)

        # Variation file
        with open(os.path.join(ep_dir, "variation.txt"), "w") as f:
            f.write(str(self.variation))

        # Per-task instructions (create/update)
        instr_path = os.path.join(task_dir, "instructions.json")
        if os.path.exists(instr_path):
            with open(instr_path) as f:
                instr = json.load(f)
        else:
            instr = {}
        var_key = str(self.variation)
        if var_key not in instr:
            instr[var_key] = []
        if self.instruction not in instr[var_key]:
            instr[var_key].append(self.instruction)
        with open(instr_path, "w") as f:
            json.dump(instr, f, indent=2)

        self.get_logger().info(f"Saved episode to {ep_dir} ({n_frames} frames)")

    # ===================== Services =====================

    def _srv_start(self, request, response):
        ok = self.start_episode()
        response.success = ok
        response.message = "Recording started" if ok else "Failed to start"
        return response

    def _srv_stop(self, request, response):
        ok = self.stop_episode()
        response.success = ok
        response.message = "Episode saved" if ok else "Failed to stop"
        return response

    # ===================== Keyboard =====================

    def _keyboard_poll(self):
        """Non-blocking keyboard check."""
        import sys
        import select
        if select.select([sys.stdin], [], [], 0.0)[0]:
            line = sys.stdin.readline().strip()
            if line.lower() == "q":
                if self.recording:
                    self.stop_episode()
                self.get_logger().info("Quitting.")
                rclpy.shutdown()
            else:
                # ENTER toggles recording
                if self.recording:
                    self.stop_episode()
                else:
                    self.start_episode()

    # ===================== Helpers =====================

    def _count_existing_episodes(self):
        task_dir = os.path.join(self.output_dir, self.task_name)
        if not os.path.isdir(task_dir):
            return 0
        return len([
            d for d in os.listdir(task_dir)
            if d.startswith("episode_") and os.path.isdir(os.path.join(task_dir, d))
        ])


def main(args=None):
    rclpy.init(args=args)
    node = DemoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.recording:
            node.stop_episode()
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
