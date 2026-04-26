# tf_fixes.py: Extract and visualize front + wrist camera point clouds from a single MCAP frame.
# tf_fixes.py: Outputs PCDs in camera frame and link_base frame for calibration debugging.

"""
Extract the first time-synced frame from an MCAP bag, unproject front and
wrist camera RGBD to point clouds, and save in both camera frame and
link_base frame for visualization in Open3D or any 3D viewer.

Front camera: /front/pose topic AND TF static (link_base -> front_optical_frame)
Wrist camera: TF kinematic chain (time-synced) AND /wrist/pose topic

Usage:
    python -m data_processing.tf_fixes

Outputs:
    data_processing/debug_pcds/front_camera_frame.ply
    data_processing/debug_pcds/front_link_base.ply
    data_processing/debug_pcds/wrist_camera_frame.ply
    data_processing/debug_pcds/wrist_link_base.ply
"""

from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

MCAP_FILE = Path('data/xarm/default_task_fixed/episode_0_bag/episode_0_bag_0.mcap')
OUTPUT_DIR = Path('data_processing/debug_pcds')


# ---------------------------------------------------------------------------
# MCAP helpers
# ---------------------------------------------------------------------------

def read_frame(mcap_path, frame_idx=0):
    """Read the Nth time-synced set of messages from the bag.

    Uses /front/color/image_raw as the reference topic to count frames.
    frame_idx=0 is the first frame, frame_idx=50 is the 50th, etc.

    Returns:
        collected: dict of topic -> (timestamp, decoded_msg) for that frame
        tf_static: dict of (parent, child) -> (trans, quat)
        tf_dynamic: dict of (parent, child) -> [(timestamp, trans, quat), ...]
        total_frames: total number of frames seen for the reference topic
    """
    ref_topic = '/front/color/image_raw'
    single_topics = {
        '/front/color/image_raw', '/front/depth/image_raw',
        '/front/camera_info', '/front/pose',
        '/wrist/color/image_raw', '/wrist/depth/image_raw',
        '/wrist/camera_info', '/wrist/pose',
    }

    # First pass: collect ALL TF data and find the Nth reference timestamp
    tf_static = {}
    tf_dynamic = {}
    ref_timestamps = []  # all timestamps for the reference topic
    all_messages = {}    # topic -> [(timestamp, msg), ...]

    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            t_sec = message.log_time / 1e9
            msg = TYPESTORE.deserialize_cdr(message.data, schema.name)

            if channel.topic == '/tf_static':
                for t in msg.transforms:
                    tr = t.transform.translation
                    rot = t.transform.rotation
                    tf_static[(t.header.frame_id, t.child_frame_id)] = (
                        np.array([tr.x, tr.y, tr.z], np.float32),
                        np.array([rot.x, rot.y, rot.z, rot.w], np.float32),
                    )
                continue

            if channel.topic == '/tf':
                for t in msg.transforms:
                    key = (t.header.frame_id, t.child_frame_id)
                    tr = t.transform.translation
                    rot = t.transform.rotation
                    if key not in tf_dynamic:
                        tf_dynamic[key] = []
                    tf_dynamic[key].append((
                        t_sec,
                        np.array([tr.x, tr.y, tr.z], np.float32),
                        np.array([rot.x, rot.y, rot.z, rot.w], np.float32),
                    ))
                continue

            if channel.topic in single_topics:
                if channel.topic == ref_topic:
                    ref_timestamps.append(t_sec)
                if channel.topic not in all_messages:
                    all_messages[channel.topic] = []
                all_messages[channel.topic].append((t_sec, msg))

    total_frames = len(ref_timestamps)
    if frame_idx >= total_frames:
        raise ValueError(f"frame_idx={frame_idx} but only {total_frames} frames in bag")

    target_time = ref_timestamps[frame_idx]

    # Pick the closest message to target_time for each topic
    collected = {}
    for topic, msgs in all_messages.items():
        closest = min(msgs, key=lambda x: abs(x[0] - target_time))
        collected[topic] = closest

    return collected, tf_static, tf_dynamic, total_frames


def read_first_frame(mcap_path):
    """Read the first frame (backward compat wrapper)."""
    collected, tf_static, tf_dynamic, _ = read_frame(mcap_path, frame_idx=0)
    return collected, tf_static, tf_dynamic


def make_mat(trans, quat):
    """Build a 4x4 homogeneous matrix from translation + quaternion (xyzw)."""
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = Rotation.from_quat(quat).as_matrix()
    m[:3, 3] = trans
    return m


def lookup_dynamic_tf(tf_dynamic, pair, target_time):
    """Find the closest dynamic TF to target_time."""
    return min(tf_dynamic[pair], key=lambda x: abs(x[0] - target_time))[1:]


def compute_world_T_wrist(tf_static, tf_dynamic, target_time):
    """Compute link_base -> wrist_optical_frame at a specific timestamp.

    Dynamic joints (link_base->link1->...->link6) are looked up at target_time.
    Static links (link6->ft_adapter->ft_sensor->link_eef->wrist_optical) are fixed.
    """
    dynamic_links = [
        ('link_base', 'link1'), ('link1', 'link2'), ('link2', 'link3'),
        ('link3', 'link4'), ('link4', 'link5'), ('link5', 'link6'),
    ]
    static_links = [
        ('link6', 'link_ft_adapter'), ('link_ft_adapter', 'link_ft_sensor'),
        ('link_ft_sensor', 'link_eef'), ('link_eef', 'wrist_optical_frame'),
    ]

    T = np.eye(4, dtype=np.float32)
    for pair in dynamic_links:
        trans, quat = lookup_dynamic_tf(tf_dynamic, pair, target_time)
        T = T @ make_mat(trans, quat)
    for pair in static_links:
        trans, quat = tf_static[pair]
        T = T @ make_mat(trans, quat)
    return T


def decode_rgb(msg):
    h, w = msg.height, msg.width
    raw = np.frombuffer(msg.data, np.uint8)
    if msg.encoding in ('bgr8', 'BGR8'):
        return cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
    return raw.reshape(h, w, -1)[:, :, :3]


def decode_depth(msg):
    h, w = msg.height, msg.width
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, np.float32).reshape(h, w)
    return np.frombuffer(msg.data, np.uint16).reshape(h, w).astype(np.float32) / 1000.0


def decode_K(msg):
    return np.array(msg.k, np.float32).reshape(3, 3)


def decode_pose_mat(msg):
    p, q = msg.pose.position, msg.pose.orientation
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    m[:3, 3] = [p.x, p.y, p.z]
    return m


# ---------------------------------------------------------------------------
# Unprojection
# ---------------------------------------------------------------------------

def unproject_to_camera_frame(depth, K):
    """Unproject depth map to 3D points in the camera optical frame.

    Returns (N, 3) xyz and (N,) valid mask indices into the flattened image.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    z = depth.flatten()
    x = (u.flatten() - K[0, 2]) * z / K[0, 0]
    y = (v.flatten() - K[1, 2]) * z / K[1, 1]

    pts = np.stack([x, y, z], axis=1)
    valid = (z > 0.01) & np.isfinite(pts).all(1)
    return pts[valid], valid


def transform_points(pts, T):
    """Apply a 4x4 rigid transform to (N, 3) points."""
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


# ---------------------------------------------------------------------------
# PLY export (no Open3D dependency needed for writing)
# ---------------------------------------------------------------------------

def save_ply(path, pts, colors=None):
    """Save a point cloud as a PLY file.

    pts: (N, 3) float32
    colors: (N, 3) uint8 or None
    """
    N = len(pts)
    has_color = colors is not None
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
    )
    if has_color:
        header += (
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
        )
    header += "end_header\n"

    with open(path, 'w') as f:
        f.write(header)
        for i in range(N):
            line = f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f}"
            if has_color:
                line += f" {colors[i,0]} {colors[i,1]} {colors[i,2]}"
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Reading {MCAP_FILE}")
    collected, tf_static, tf_dynamic = read_first_frame(MCAP_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Front camera (static transform: link_base -> front_optical_frame)
    # ==================================================================
    print("\n=== Front camera ===")
    _, front_rgb_msg = collected['/front/color/image_raw']
    _, front_depth_msg = collected['/front/depth/image_raw']
    _, front_info_msg = collected['/front/camera_info']
    _, front_pose_msg = collected['/front/pose']

    front_rgb = decode_rgb(front_rgb_msg)
    front_depth = decode_depth(front_depth_msg)
    front_K = decode_K(front_info_msg)

    print(f"  Image: {front_rgb.shape[1]}x{front_rgb.shape[0]}")
    print(f"  Depth: {front_depth[front_depth > 0].min():.3f} - {front_depth.max():.3f} m")
    print(f"  K: fx={front_K[0,0]:.1f}  fy={front_K[1,1]:.1f}  cx={front_K[0,2]:.1f}  cy={front_K[1,2]:.1f}")

    front_pts_cam, front_valid = unproject_to_camera_frame(front_depth, front_K)
    front_colors = front_rgb.reshape(-1, 3)[front_valid]
    print(f"  Points (camera frame): {len(front_pts_cam)}")

    # --- /front/pose topic ---
    world_T_front_pose = decode_pose_mat(front_pose_msg)
    print(f"  /front/pose: t=({world_T_front_pose[0,3]:.4f}, {world_T_front_pose[1,3]:.4f}, {world_T_front_pose[2,3]:.4f})")
    front_pose_euler = Rotation.from_matrix(world_T_front_pose[:3, :3]).as_euler('xyz', degrees=True)
    print(f"               euler=({front_pose_euler[0]:.1f}, {front_pose_euler[1]:.1f}, {front_pose_euler[2]:.1f})")

    front_pts_base_pose = transform_points(front_pts_cam, world_T_front_pose)
    print(f"  Points (/front/pose → link_base): Z range [{front_pts_base_pose[:,2].min():.3f}, {front_pts_base_pose[:,2].max():.3f}]")

    # --- TF static: link_base -> front_optical_frame ---
    if ('link_base', 'front_optical_frame') in tf_static:
        tf_t, tf_q = tf_static[('link_base', 'front_optical_frame')]
        world_T_front_tf = make_mat(tf_t, tf_q)
        tf_euler = Rotation.from_quat(tf_q).as_euler('xyz', degrees=True)
        print(f"  TF static:   t=({tf_t[0]:.4f}, {tf_t[1]:.4f}, {tf_t[2]:.4f})")
        print(f"               euler=({tf_euler[0]:.1f}, {tf_euler[1]:.1f}, {tf_euler[2]:.1f})")

        front_pts_base_tf = transform_points(front_pts_cam, world_T_front_tf)
        print(f"  Points (TF static → link_base): Z range [{front_pts_base_tf[:,2].min():.3f}, {front_pts_base_tf[:,2].max():.3f}]")

        save_ply(OUTPUT_DIR / 'front_link_base_tf.ply', front_pts_base_tf, front_colors)
        print(f"  Saved: front_link_base_tf.ply (from TF static)")

    save_ply(OUTPUT_DIR / 'front_camera_frame.ply', front_pts_cam, front_colors)
    save_ply(OUTPUT_DIR / 'front_link_base.ply', front_pts_base_pose, front_colors)
    print(f"  Saved: front_camera_frame.ply, front_link_base.ply (from /front/pose)")

    # ==================================================================
    # Wrist camera (dynamic transform via TF kinematic chain)
    # ==================================================================
    print("\n=== Wrist camera ===")
    wrist_t, wrist_rgb_msg = collected['/wrist/color/image_raw']
    _, wrist_depth_msg = collected['/wrist/depth/image_raw']
    _, wrist_info_msg = collected['/wrist/camera_info']
    _, wrist_pose_msg = collected['/wrist/pose']

    wrist_rgb = decode_rgb(wrist_rgb_msg)
    wrist_depth = decode_depth(wrist_depth_msg)
    wrist_K = decode_K(wrist_info_msg)

    print(f"  Image: {wrist_rgb.shape[1]}x{wrist_rgb.shape[0]}")
    print(f"  Depth: {wrist_depth[wrist_depth > 0].min():.3f} - {wrist_depth.max():.3f} m")
    print(f"  K: fx={wrist_K[0,0]:.1f}  fy={wrist_K[1,1]:.1f}  cx={wrist_K[0,2]:.1f}  cy={wrist_K[1,2]:.1f}")

    wrist_pts_cam, wrist_valid = unproject_to_camera_frame(wrist_depth, wrist_K)
    wrist_colors = wrist_rgb.reshape(-1, 3)[wrist_valid]
    print(f"  Points (camera frame): {len(wrist_pts_cam)}")

    # Dynamic transform: time-synced TF chain at the wrist depth timestamp
    world_T_wrist = compute_world_T_wrist(tf_static, tf_dynamic, wrist_t)
    print(f"  TF chain (t={wrist_t:.3f}): t=({world_T_wrist[0,3]:.4f}, {world_T_wrist[1,3]:.4f}, {world_T_wrist[2,3]:.4f})")

    # Show /wrist/pose for comparison
    world_T_wrist_pose = decode_pose_mat(wrist_pose_msg)
    print(f"  /wrist/pose:            t=({world_T_wrist_pose[0,3]:.4f}, {world_T_wrist_pose[1,3]:.4f}, {world_T_wrist_pose[2,3]:.4f})")

    # Show the static link_eef -> wrist_optical_frame for reference
    eef_t, eef_q = tf_static[('link_eef', 'wrist_optical_frame')]
    eef_euler = Rotation.from_quat(eef_q).as_euler('xyz', degrees=True)
    print(f"  link_eef->wrist TF: t=({eef_t[0]:.5f}, {eef_t[1]:.5f}, {eef_t[2]:.5f})  euler=({eef_euler[0]:.1f}, {eef_euler[1]:.1f}, {eef_euler[2]:.1f})")

    # --- TF chain variant ---
    wrist_pts_base_tf = transform_points(wrist_pts_cam, world_T_wrist)
    print(f"  Points (TF chain → link_base): Z range [{wrist_pts_base_tf[:,2].min():.3f}, {wrist_pts_base_tf[:,2].max():.3f}]")

    # --- /wrist/pose variant ---
    wrist_pts_base_pose = transform_points(wrist_pts_cam, world_T_wrist_pose)
    print(f"  Points (/wrist/pose → link_base): Z range [{wrist_pts_base_pose[:,2].min():.3f}, {wrist_pts_base_pose[:,2].max():.3f}]")

    save_ply(OUTPUT_DIR / 'wrist_camera_frame.ply', wrist_pts_cam, wrist_colors)
    save_ply(OUTPUT_DIR / 'wrist_link_base.ply', wrist_pts_base_tf, wrist_colors)
    save_ply(OUTPUT_DIR / 'wrist_link_base_pose.ply', wrist_pts_base_pose, wrist_colors)
    print(f"  Saved: wrist_link_base.ply (TF chain), wrist_link_base_pose.ply (/wrist/pose)")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n=== Output files ===")
    for f in sorted(OUTPUT_DIR.glob('*.ply')):
        print(f"  {f}")
    print("\nOverlay front_link_base.ply + wrist_link_base.ply in a 3D viewer")
    print("to check if the two point clouds align in world frame.")


if __name__ == '__main__':
    main()
