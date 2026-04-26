# bag_to_episodes.py: Extract ROS2 bag files into episode directories for VLA training.
# bag_to_episodes.py: Handles mixed camera setups (Azure Kinect + RealSense D435i).

"""
Convert snippeted ROS2 bag files into episode directories compatible with
XArmDataset for direct training (no zarr conversion needed).

Expected bag structure:
    <bag_dir>/
        <task_name>/
            episode_0.mcap (or episode_0/ for .db3 format)
            episode_1.mcap
            ...
            instructions.json  (optional, {"0": ["instruction text"]})

Output structure:
    <output_dir>/
        <task_name>/
            episode_0/
                rgb/front_0000.png, wrist_0000.png
                depth/front_0000.png, wrist_0000.png
                eef_states.npy          # (T, 8)
                camera_extrinsics.npy   # (ncam, 4, 4)
                camera_intrinsics.npy   # (ncam, 3, 3)
            instructions.json

Usage:
    python -m data_generation.bag_to_episodes --config configs/extraction.yaml

    Override any config value via CLI:
    python -m data_generation.bag_to_episodes --config configs/extraction.yaml \
        --bag_dir /other/path --target_hz 5.0
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from mcap.reader import make_reader as _make_mcap_reader
from rosbags.typesys import Stores, get_typestore

from utils.config import load_yaml_config, flatten_config


IM_SIZE = 256
# Episode depth PNGs are stored as uint16 millimetres.
# xarm_to_zarr.py divides by the same value to convert back to metres.
DEPTH_MM_SCALE = 1000.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract ROS2 bags into episode directories"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to extraction YAML config"
    )
    # Allow CLI overrides for any flat config key
    parser.add_argument("--bag_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--target_hz", type=float, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--sync_slop", type=float, default=None)
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode indices to extract (e.g. '0,5,10')")
    return parser.parse_args()


def load_extraction_config(args):
    """Load YAML config and apply CLI overrides."""
    config = load_yaml_config(args.config)
    # Apply direct CLI overrides
    for key in ['bag_dir', 'output_dir', 'target_hz', 'image_size',
                'sync_slop']:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            config[key] = cli_val
    return config


def _nanosec_to_sec(ns):
    return ns / 1e9


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


def _decode_image(msg, typestore):
    """Decode a sensor_msgs/Image into numpy array."""
    encoding = msg.encoding
    h, w = msg.height, msg.width
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ('rgb8', 'RGB8'):
        return data.reshape(h, w, 3)
    elif encoding in ('bgr8', 'BGR8'):
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
    elif encoding in ('16UC1',):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    elif encoding in ('32FC1',):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
    elif encoding in ('mono8',):
        return data.reshape(h, w)
    elif encoding in ('rgba8', 'RGBA8'):
        return data.reshape(h, w, 4)[:, :, :3]
    elif encoding in ('bgra8', 'BGRA8'):
        img = data.reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")


def _decode_pose(msg):
    """Extract [x, y, z, qx, qy, qz, qw] from PoseStamped."""
    p = msg.pose.position
    q = msg.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)


def _decode_gripper(msg):
    """Extract gripper state as float [0, 1]."""
    # Support Float64, Float32, or Bool
    if hasattr(msg, 'data'):
        val = float(msg.data)
        # Normalize to [0, 1] if needed
        if val > 1.0:
            val = val / 255.0
        return np.float32(val)
    return np.float32(0.0)


def _decode_camera_info(msg):
    """Extract intrinsic matrix from CameraInfo."""
    K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
    return K


def _resize_image(img, target_size):
    """Resize image to target_size x target_size."""
    if img.shape[0] == target_size and img.shape[1] == target_size:
        return img
    interp = cv2.INTER_NEAREST if img.dtype == np.uint16 else cv2.INTER_AREA
    return cv2.resize(img, (target_size, target_size), interpolation=interp)


def _make_mat(trans, quat):
    """Build a 4x4 matrix from translation + quaternion (xyzw)."""
    from scipy.spatial.transform import Rotation as R
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R.from_quat(quat).as_matrix()
    m[:3, 3] = trans
    return m


def _read_all_tf(mcap_path, typestore):
    """Read all TF data from an MCAP file.

    Returns:
        tf_static: dict of (parent, child) -> (trans, quat)
        tf_dynamic: dict of (parent, child) -> [(timestamp, trans, quat), ...]
    """
    tf_static = {}
    tf_dynamic = {}
    with open(mcap_path, 'rb') as f:
        reader = _make_mcap_reader(f)
        for schema, channel, message in reader.iter_messages():
            if channel.topic not in ('/tf_static', '/tf'):
                continue
            msg = typestore.deserialize_cdr(message.data, schema.name)
            t_sec = message.log_time / 1e9
            for t in msg.transforms:
                parent = t.header.frame_id.strip('/')
                child = t.child_frame_id.strip('/')
                tr = t.transform.translation
                rot = t.transform.rotation
                trans = np.array([tr.x, tr.y, tr.z], np.float32)
                quat = np.array([rot.x, rot.y, rot.z, rot.w], np.float32)

                if channel.topic == '/tf_static':
                    tf_static[(parent, child)] = (trans, quat)
                else:
                    key = (parent, child)
                    if key not in tf_dynamic:
                        tf_dynamic[key] = []
                    tf_dynamic[key].append((t_sec, trans, quat))
    return tf_static, tf_dynamic


def _lookup_dynamic_tf(tf_dynamic, pair, target_time):
    """Find the closest dynamic TF to target_time. Returns (trans, quat, dt)."""
    entry = min(tf_dynamic[pair], key=lambda x: abs(x[0] - target_time))
    dt = abs(entry[0] - target_time)
    return entry[1], entry[2], dt


def _compute_tf_chain(tf_static, tf_dynamic, chain, target_time):
    """Compute a chained transform at a specific timestamp.

    chain: list of (parent, child) pairs. Each pair is looked up in
           tf_dynamic first (time-synced), then tf_static.
    Returns (4x4 matrix, max_dt) where max_dt is the worst time sync delta.
    """
    T = np.eye(4, dtype=np.float32)
    max_dt = 0.0
    for pair in chain:
        if pair in tf_dynamic:
            trans, quat, dt = _lookup_dynamic_tf(tf_dynamic, pair, target_time)
            max_dt = max(max_dt, dt)
        elif pair in tf_static:
            trans, quat = tf_static[pair]
        else:
            raise KeyError(f"TF pair {pair} not found in static or dynamic")
        T = T @ _make_mat(trans, quat)
    return T, max_dt


def _compute_static_extrinsic(tf_static, cam_frame, world_frame):
    """Look up a direct static transform world_frame -> cam_frame."""
    if (world_frame, cam_frame) in tf_static:
        trans, quat = tf_static[(world_frame, cam_frame)]
        return _make_mat(trans, quat)
    if (cam_frame, world_frame) in tf_static:
        trans, quat = tf_static[(cam_frame, world_frame)]
        return np.linalg.inv(_make_mat(trans, quat))
    return None


def _collect_messages_by_topic(mcap_path, typestore, topics):
    """Read all messages from given topics using mcap reader.
    Returns dict of {topic: [(time_sec, decoded_msg)]}."""
    result = {t: [] for t in topics}
    with open(mcap_path, 'rb') as f:
        reader = _make_mcap_reader(f)
        for schema, channel, message in reader.iter_messages():
            if channel.topic in result:
                msg = typestore.deserialize_cdr(message.data, schema.name)
                t_sec = message.log_time / 1e9
                result[channel.topic].append((t_sec, msg))
    return result


def process_bag(bag_path, config):
    """
    Process a single ROS2 bag file into episode data arrays.

    Returns dict with rgb, depth, eef_states, extrinsics, intrinsics
    or None if the bag is invalid.
    """
    cameras = config['cameras']
    cam_names = list(cameras.keys())
    ncam = len(cam_names)
    im_size = config.get('image_size', IM_SIZE)
    target_hz = config.get('target_hz', 10.0)
    sync_slop = config.get('sync_slop', 0.05)
    eef_topic = config['robot']['eef_topic']
    gripper_topic = config['robot']['gripper_topic']
    world_frame = config['robot']['world_frame']

    # Collect all topics we need
    all_topics = [eef_topic, gripper_topic]
    for cam_name, cam_cfg in cameras.items():
        all_topics.extend([
            cam_cfg['rgb_topic'],
            cam_cfg['depth_topic'],
            cam_cfg['camera_info_topic'],
        ])

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    bag_path = Path(bag_path)

    # Find the MCAP file inside the bag directory
    if bag_path.is_dir():
        mcap_files = list(bag_path.glob('*.mcap'))
        if not mcap_files:
            print(f"  Skipping {bag_path}: no MCAP files found")
            return None
        mcap_path = mcap_files[0]
    else:
        mcap_path = bag_path

    # Read all messages using mcap reader (supports v9/Jazzy bags)
    msgs = _collect_messages_by_topic(mcap_path, typestore, all_topics)

    # Get EEF trajectory as the time reference
    eef_msgs = msgs[eef_topic]
    gripper_msgs = msgs[gripper_topic]

    if len(eef_msgs) < 2:
        print(f"  Skipping {bag_path}: too few EEF messages ({len(eef_msgs)})")
        return None

    # Determine time range and resample at target_hz
    t_start = eef_msgs[0][0]
    t_end = eef_msgs[-1][0]
    dt = 1.0 / target_hz
    sample_times = np.arange(t_start, t_end, dt)

    if len(sample_times) < 2:
        print(f"  Skipping {bag_path}: duration too short")
        return None

    # Extract camera intrinsics, scaled to im_size x im_size.
    # CameraInfo K is for the original resolution; images are resized
    # to im_size so K must be scaled to match.
    intrinsics = np.zeros((ncam, 3, 3), dtype=np.float32)
    for i, cam_name in enumerate(cam_names):
        info_msgs = msgs[cameras[cam_name]['camera_info_topic']]
        rgb_msgs = msgs[cameras[cam_name]['rgb_topic']]
        if info_msgs:
            K = _decode_camera_info(info_msgs[0][1])
            if rgb_msgs:
                orig_w, orig_h = rgb_msgs[0][1].width, rgb_msgs[0][1].height
                K[0, 0] *= im_size / orig_w   # fx
                K[1, 1] *= im_size / orig_h   # fy
                K[0, 2] *= im_size / orig_w   # cx
                K[1, 2] *= im_size / orig_h   # cy
            intrinsics[i] = K
        else:
            print(f"  Warning: no CameraInfo for {cam_name}")

    # Read all TF data (static + dynamic) from the bag
    tf_static, tf_dynamic = _read_all_tf(mcap_path, typestore)

    # Build per-camera TF chain configs.
    # tf_chain: list of (parent, child) pairs from world_frame to cam_frame.
    # If a direct static TF exists, the chain is just that single pair.
    # If the camera has a tf_chain config, use that for dynamic resolution.
    cam_chains = {}
    cam_static_ext = {}
    has_dynamic_cam = False
    for cam_name in cam_names:
        cam_cfg = cameras[cam_name]
        cam_frame = cam_cfg['tf_frame']
        if 'tf_chain' in cam_cfg:
            # Explicit chain specified in config (for eye-in-hand cameras)
            cam_chains[cam_name] = [tuple(pair) for pair in cam_cfg['tf_chain']]
            has_dynamic_cam = True
        else:
            # Try direct static TF lookup
            ext = _compute_static_extrinsic(tf_static, cam_frame, world_frame)
            if ext is not None:
                cam_static_ext[cam_name] = ext
            else:
                print(f"  Warning: no static TF for {cam_name} "
                      f"({cam_frame} -> {world_frame}), using identity")
                cam_static_ext[cam_name] = np.eye(4, dtype=np.float32)

    # Resample all data at target_hz
    rgbs = []
    depths = []
    eef_states = []
    per_frame_ext_list = []
    tf_sync_deltas = []  # time sync deltas for TF chain lookups

    for t_sample in sample_times:
        # EEF pose
        eef_msg = _find_closest_msg(t_sample, eef_msgs, sync_slop * 2)
        grip_msg = _find_closest_msg(t_sample, gripper_msgs, sync_slop * 2)
        if eef_msg is None:
            continue

        pose = _decode_pose(eef_msg)
        gripper = _decode_gripper(grip_msg) if grip_msg is not None else np.float32(0.0)
        eef_state = np.concatenate([pose, [gripper]])
        eef_states.append(eef_state)

        # Camera images and per-frame extrinsics
        frame_rgbs = []
        frame_depths = []
        frame_ext = np.zeros((ncam, 4, 4), dtype=np.float32)
        valid = True

        for i, cam_name in enumerate(cam_names):
            cam_cfg = cameras[cam_name]
            rgb_msg = _find_closest_msg(
                t_sample, msgs[cam_cfg['rgb_topic']], sync_slop
            )
            depth_msg = _find_closest_msg(
                t_sample, msgs[cam_cfg['depth_topic']], sync_slop
            )

            if rgb_msg is None or depth_msg is None:
                valid = False
                break

            rgb = _decode_image(rgb_msg, typestore)
            depth = _decode_image(depth_msg, typestore)

            rgb = _resize_image(rgb, im_size)
            depth = _resize_image(depth, im_size)

            frame_rgbs.append(rgb)
            frame_depths.append(depth)

            # Extrinsics: TF chain (dynamic) or static lookup
            if cam_name in cam_chains:
                mat, dt = _compute_tf_chain(
                    tf_static, tf_dynamic, cam_chains[cam_name], t_sample)
                frame_ext[i] = mat
                tf_sync_deltas.append(dt)
            else:
                frame_ext[i] = cam_static_ext[cam_name]

        if not valid:
            eef_states.pop()
            continue

        rgbs.append(frame_rgbs)
        depths.append(frame_depths)
        per_frame_ext_list.append(frame_ext)

    if len(eef_states) < 2:
        print(f"  Skipping {bag_path}: too few synchronized frames ({len(eef_states)})")
        return None

    eef_states = np.stack(eef_states)  # (T, 8)

    # Time sync validation for TF chain lookups
    if tf_sync_deltas:
        max_dt = max(tf_sync_deltas)
        mean_dt = np.mean(tf_sync_deltas)
        warn_threshold = sync_slop
        if max_dt > warn_threshold:
            print(f"  WARNING: TF sync max_dt={max_dt*1000:.1f}ms > slop={warn_threshold*1000:.0f}ms "
                  f"(mean={mean_dt*1000:.1f}ms)")
        else:
            print(f"  TF sync: max_dt={max_dt*1000:.1f}ms, mean={mean_dt*1000:.1f}ms (OK)")

    # Extrinsics: (T, ncam, 4, 4) if any camera has dynamic TF chain, else (ncam, 4, 4)
    if has_dynamic_cam:
        extrinsics = np.stack(per_frame_ext_list)  # (T, ncam, 4, 4)
    else:
        extrinsics = np.stack([cam_static_ext[c] for c in cam_names])  # (ncam, 4, 4)

    return {
        'rgbs': rgbs,           # list of T x [ncam images]
        'depths': depths,       # list of T x [ncam depth maps]
        'eef_states': eef_states,
        'extrinsics': extrinsics,
        'intrinsics': intrinsics,
    }


def save_episode(data, episode_dir, cam_names):
    """Save extracted episode data to the standard directory structure."""
    os.makedirs(episode_dir, exist_ok=True)
    rgb_dir = os.path.join(episode_dir, 'rgb')
    depth_dir = os.path.join(episode_dir, 'depth')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    T = len(data['rgbs'])
    for t in range(T):
        for c, cam_name in enumerate(cam_names):
            # Save RGB as PNG
            rgb = data['rgbs'][t][c]
            cv2.imwrite(
                os.path.join(rgb_dir, f'{cam_name}_{t:04d}.png'),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )
            # Save depth as uint16 PNG in millimetres.
            # Convention: episode depth PNGs are always uint16 mm.
            # xarm_to_zarr.py divides by 1000 to convert back to metres.
            depth = data['depths'][t][c]
            if depth.dtype == np.float32:
                depth_mm = (depth * DEPTH_MM_SCALE).astype(np.uint16)
            elif depth.dtype == np.uint16:
                depth_mm = depth
            else:
                depth_mm = depth.astype(np.uint16)
            cv2.imwrite(
                os.path.join(depth_dir, f'{cam_name}_{t:04d}.png'),
                depth_mm
            )

    np.save(os.path.join(episode_dir, 'eef_states.npy'), data['eef_states'])
    np.save(os.path.join(episode_dir, 'camera_extrinsics.npy'), data['extrinsics'])
    np.save(os.path.join(episode_dir, 'camera_intrinsics.npy'), data['intrinsics'])


def find_bag_files(task_dir):
    """Find all bag files/directories in a task directory."""
    bags = []
    for entry in sorted(os.listdir(task_dir)):
        full_path = os.path.join(task_dir, entry)
        # .mcap files at top level
        if entry.endswith('.mcap'):
            bags.append(full_path)
        elif os.path.isdir(full_path):
            # Directories containing .mcap files (e.g. episode_0_bag/)
            mcap_files = [f for f in os.listdir(full_path) if f.endswith('.mcap')]
            if mcap_files:
                bags.append(full_path)
            # .db3 directories (ROS2 bag format)
            db3_files = [f for f in os.listdir(full_path) if f.endswith('.db3')]
            if db3_files:
                bags.append(full_path)
    return bags


def main():
    args = parse_arguments()
    config = load_extraction_config(args)

    bag_dir = config['bag_dir']
    output_dir = config['output_dir']
    tasks = config['tasks']
    cam_names = list(config['cameras'].keys())

    total_episodes = 0

    for task in tasks:
        task_bag_dir = os.path.join(bag_dir, task)
        if not os.path.isdir(task_bag_dir):
            print(f"Warning: no bag directory for task '{task}' at {task_bag_dir}")
            continue

        bags = find_bag_files(task_bag_dir)
        print(f"Task '{task}': found {len(bags)} bag files")

        # Filter to specific episodes if requested
        episode_filter = None
        if args.episodes is not None:
            episode_filter = set(int(x) for x in args.episodes.split(','))
            print(f"  Filtering to episodes: {sorted(episode_filter)}")

        task_output_dir = os.path.join(output_dir, task)
        os.makedirs(task_output_dir, exist_ok=True)

        for ep_idx, bag_path in enumerate(tqdm(bags, desc=f"  {task}")):
            if episode_filter is not None and ep_idx not in episode_filter:
                continue

            episode_dir = os.path.join(task_output_dir, f'episode_{ep_idx}')

            if os.path.exists(episode_dir):
                print(f"    Skipping {episode_dir}: already exists")
                total_episodes += 1
                continue

            data = process_bag(bag_path, config)
            if data is None:
                continue

            save_episode(data, episode_dir, cam_names)
            total_episodes += 1
            print(f"    Saved episode_{ep_idx}: {len(data['eef_states'])} frames")

        # Copy instructions.json if present in bag dir
        instr_src = os.path.join(task_bag_dir, 'instructions.json')
        instr_dst = os.path.join(task_output_dir, 'instructions.json')
        if os.path.exists(instr_src) and not os.path.exists(instr_dst):
            shutil.copy2(instr_src, instr_dst)
        elif not os.path.exists(instr_dst):
            # Create default instruction from task name
            default_instr = {
                "0": [task.replace("_", " ")]
            }
            with open(instr_dst, 'w') as f:
                json.dump(default_instr, f, indent=2)
            print(f"    Created default instructions.json for {task}")

    print(f"\nExtracted {total_episodes} episodes to {output_dir}")


if __name__ == '__main__':
    main()
