"""
Convert xArm episode directories to Zarr format for 3D FlowMatch Actor training.

Expected input directory structure:
    <root>/
        <task_name>/
            episode_0/
                rgb/<cam_name>_0000.png       # 256x256 RGB images
                depth/<cam_name>_0000.png     # 16-bit PNG depth (millimeters)
                eef_states.npy                # (T, 8) float32: [x,y,z, qw,qx,qy,qz, gripper]
                camera_extrinsics.npy         # (ncam, 4, 4) float32
                camera_intrinsics.npy         # (ncam, 3, 3) float32
                camera_names.json             # ["wrist", "front"]
            instructions.json                 # {"0": ["pick up the red cup", ...]}

Output: train.zarr/ and val.zarr/ for XArmDataset.

Usage:
    python -m data_processing.xarm_to_zarr --config configs/zarr.yaml
"""

import argparse
import json
import os

import numpy as np
import zarr
from numcodecs import Blosc
from PIL import Image
from tqdm import tqdm

from data_processing.rlbench_utils import (
    interpolate_trajectory,
    quat_to_euler_np,
    euler_to_quat_np,
)


IM_SIZE = 256
# Episode depth PNGs are uint16 millimetres (written by bag_to_episodes.py).
# Divide by this value to convert to metres for the zarr.
DEPTH_MM_SCALE = 1000.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert real arm demos to zarr for 3D FlowMatch Actor"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. CLI args override config values."
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Root directory containing task folders with episodes"
    )
    parser.add_argument(
        "--tgt", type=str, default=None,
        help="Output directory for zarr files"
    )
    parser.add_argument(
        "--cameras", type=str, nargs="+", default=None,
        help="Camera names, matching filenames in rgb/ and depth/ folders"
    )
    parser.add_argument(
        "--nhand", type=int, default=None, choices=[1, 2],
        help="Number of arms (1=single, 2=bimanual)"
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None,
        help="List of task folder names"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=None,
        help="Fraction of episodes to use for validation"
    )
    parser.add_argument(
        "--trajectory_length", type=int, default=None,
        help="Action trajectory length. 1=keypose only, >1=interpolated trajectory"
    )
    parser.add_argument(
        "--num_history", type=int, default=None,
        help="Number of proprioception history steps"
    )
    parser.add_argument(
        "--keyframe_gripper_change", action="store_true", default=None,
        help="Auto-detect keyframes from gripper state changes instead of using all timesteps"
    )
    parser.add_argument(
        "--keyframe_velocity_threshold", type=float, default=None,
        help="Velocity threshold for keyframe detection (m/s)"
    )
    parser.add_argument(
        "--keyframe_distance_threshold", type=float, default=None,
        help="EEF travel distance per keyframe segment in metres "
             "(e.g. 0.0127 = 0.5 inch). When set, overrides gripper/velocity detection."
    )
    parser.add_argument(
        "--input_quat_format", type=str, default=None,
        choices=["wxyz", "xyzw"],
        help="Quaternion format in eef_states.npy (required)"
    )
    return parser.parse_args()


def _resolve_args(args):
    """Merge YAML config with CLI args, CLI takes precedence."""
    # Defaults when neither config nor CLI provides a value
    defaults = {
        'root': '',
        'tgt': '',
        'cameras': ['front', 'wrist'],
        'nhand': 1,
        'tasks': [],
        'val_ratio': 0.1,
        'trajectory_length': 1,
        'num_history': 3,
        'keyframe_gripper_change': False,
        'keyframe_velocity_threshold': 0.01,
        'keyframe_distance_threshold': None,
        'input_quat_format': None,  # required: 'wxyz' or 'xyzw'
    }

    if args.config is not None:
        from utils.config import load_yaml_config
        config = load_yaml_config(args.config)
        # Apply config values as base
        for key, default in defaults.items():
            if key in config:
                defaults[key] = config[key]

    # CLI overrides (only if explicitly set, i.e., not None)
    for key in defaults:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            defaults[key] = cli_val

    # Write back to args namespace
    for key, val in defaults.items():
        setattr(args, key, val)

    # Validate required fields
    if not args.root:
        raise ValueError("--root (or config 'root') is required")
    if not args.tgt:
        raise ValueError("--tgt (or config 'tgt') is required")
    if not args.tasks:
        raise ValueError("--tasks (or config 'tasks') is required")

    return args


def load_rgb(episode_dir, cameras, timestep):
    """Load RGB images for all cameras at a given timestep. Returns (ncam, 3, H, W) uint8."""
    imgs = []
    for cam in cameras:
        path = os.path.join(episode_dir, "rgb", f"{cam}_{timestep:04d}.png")
        img = np.array(Image.open(path).resize((IM_SIZE, IM_SIZE)))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        imgs.append(img[:, :, :3].transpose(2, 0, 1))  # (3, H, W)
    return np.stack(imgs).astype(np.uint8)  # (ncam, 3, H, W)


def load_depth(episode_dir, cameras, timestep):
    """Load depth maps for all cameras at a given timestep. Returns (ncam, H, W) float16."""
    depths = []
    for cam in cameras:
        path = os.path.join(episode_dir, "depth", f"{cam}_{timestep:04d}.png")
        raw = np.array(Image.open(path))
        # Convert uint16 mm to float32 metres
        depth_m = raw.astype(np.float32) / DEPTH_MM_SCALE
        # Resize if needed
        if depth_m.shape != (IM_SIZE, IM_SIZE):
            depth_m = np.array(
                Image.fromarray(depth_m).resize((IM_SIZE, IM_SIZE), Image.NEAREST)
            )
        depths.append(depth_m.astype(np.float16))
    return np.stack(depths)  # (ncam, H, W)


def detect_keyframes(eef_states, gripper_change=True, vel_threshold=0.01,
                     distance_threshold=None):
    """
    Detect keyframes from EEF trajectory.

    Two modes (distance_threshold takes priority when set):
    - distance_threshold: place a keyframe each time EEF travels this far
      (metres) from the last keyframe. E.g. 0.0127 m = 0.5 inch. Use for
      tasks with constant gripper state where open/close events don't occur.
    - default: keyframes at gripper state changes and low-velocity stops.

    Always includes frame 0 and T-1.

    Args:
        eef_states: (T, nhand*8) float32 EEF states
        gripper_change: use gripper transitions (ignored if distance_threshold set)
        vel_threshold: stop threshold in m/step (ignored if distance_threshold set)
        distance_threshold: EEF travel per segment in metres, or None

    Returns:
        Sorted list of keyframe indices
    """
    T = len(eef_states)
    if T <= 2:
        return list(range(T))

    if eef_states.ndim == 1:
        eef_states = eef_states.reshape(T, -1)

    arm_dim = 8

    if distance_threshold is not None:
        # Distance-based: new keyframe every time cumulative EEF travel >= threshold
        keyframes = [0]
        positions = eef_states[:, :3]  # use first arm positions
        accumulated = 0.0
        for i in range(1, T):
            accumulated += np.linalg.norm(positions[i] - positions[i - 1])
            if accumulated >= distance_threshold:
                keyframes.append(i)
                accumulated = 0.0
        if keyframes[-1] != T - 1:
            keyframes.append(T - 1)
        return keyframes

    # Default: gripper changes + velocity stops
    keyframes = set([0, T - 1])
    for arm in range(max(1, eef_states.shape[1] // arm_dim)):
        start = arm * arm_dim
        positions = eef_states[:, start:start + 3]
        gripper = eef_states[:, start + 7]

        if gripper_change:
            gripper_binary = (gripper > 0.5).astype(int)
            changes = np.where(np.diff(gripper_binary) != 0)[0] + 1
            keyframes.update(changes.tolist())

        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        stopped = velocities < vel_threshold
        for i in range(1, len(stopped)):
            if stopped[i] and not stopped[i - 1]:
                keyframes.add(i + 1)

    return sorted(keyframes)


def build_proprioception(eef_states, keyframes, num_history, nhand):
    """
    Build proprioception with history from EEF states at keyframes.

    Returns (num_keyframes, num_history, nhand, 8) float32
    """
    # Get states at keyframes
    states = eef_states[keyframes]  # (K, nhand*8) or (K, 8)
    states = states.reshape(len(keyframes), nhand, 8)

    # Build history by repeating the first frame for padding
    props = []
    for i in range(len(keyframes)):
        history = []
        for h in range(num_history - 1, -1, -1):
            idx = max(0, i - h)
            history.append(states[idx])
        props.append(np.stack(history))  # (num_history, nhand, 8)

    return np.stack(props).astype(np.float32)  # (K, num_history, nhand, 8)


def build_actions(eef_states, keyframes, trajectory_length, nhand):
    """
    Build action targets from EEF states.

    For keypose_only (trajectory_length=1): action is the next keyframe's state.
    For trajectory mode: interpolate between consecutive keyframes.

    Returns (num_samples, trajectory_length, nhand, 8) float32
    """
    actions = []
    for i in range(len(keyframes) - 1):
        start_idx = keyframes[i]
        end_idx = keyframes[i + 1]

        if trajectory_length == 1:
            # Keypose only: target is the next keyframe
            action = eef_states[end_idx].reshape(1, nhand, 8)
        else:
            # Trajectory mode: interpolate the segment
            segment = eef_states[start_idx:end_idx + 1]
            segment = segment.reshape(len(segment), nhand, 8)
            interp_actions = []
            for hand in range(nhand):
                hand_traj = segment[:, hand, :]  # (seg_len, 8)
                # Convert quat to euler for smooth interpolation
                euler_traj = np.concatenate([
                    hand_traj[:, :3],
                    quat_to_euler_np(hand_traj[:, 3:7]),
                    hand_traj[:, 7:]
                ], axis=1)
                # Interpolate
                interp = interpolate_trajectory(euler_traj, trajectory_length)
                # Convert back to quat
                interp = np.concatenate([
                    interp[:, :3],
                    euler_to_quat_np(interp[:, 3:6]),
                    interp[:, 6:]
                ], axis=1)
                interp_actions.append(interp)
            action = np.stack(interp_actions, axis=1)  # (T, nhand, 8)

        actions.append(action)

    return np.stack(actions).astype(np.float32)  # (num_samples, T, nhand, 8)



def _convert_quat_wxyz_to_xyzw(eef_states):
    """Convert quaternion columns from [w,x,y,z] to [x,y,z,w] in-place."""
    # eef_states: (T, 8) with columns [x,y,z, qw,qx,qy,qz, gripper]
    # Convert to [x,y,z, qx,qy,qz,qw, gripper]
    quat_wxyz = eef_states[:, 3:7].copy()
    eef_states[:, 3] = quat_wxyz[:, 1]  # qx
    eef_states[:, 4] = quat_wxyz[:, 2]  # qy
    eef_states[:, 5] = quat_wxyz[:, 3]  # qz
    eef_states[:, 6] = quat_wxyz[:, 0]  # qw
    return eef_states


def _read_camera_names(episode_dir):
    """Read camera_names.json if present, returns list of camera names or None."""
    path = os.path.join(episode_dir, "camera_names.json")
    if os.path.exists(path):
        return json.load(open(path))
    return None


def _reorder_cameras(data_cameras, target_cameras, extrinsics, intrinsics):
    """
    Reorder camera arrays to match the target camera ordering.

    Args:
        data_cameras: list of camera names as stored in the data (from camera_names.json)
        target_cameras: list of camera names in the desired order (from config)
        extrinsics: (ncam, 4, 4) array
        intrinsics: (ncam, 3, 3) array

    Returns:
        Reordered (extrinsics, intrinsics) and a mapping for image loading
    """
    if data_cameras == target_cameras:
        return extrinsics, intrinsics, None

    # Build index mapping: target_idx -> data_idx
    cam_map = []
    for cam in target_cameras:
        if cam in data_cameras:
            cam_map.append(data_cameras.index(cam))
        else:
            raise ValueError(
                f"Camera '{cam}' not found in episode data. "
                f"Available: {data_cameras}"
            )

    reordered_ext = extrinsics[cam_map]
    reordered_intr = intrinsics[cam_map]
    return reordered_ext, reordered_intr, cam_map


def process_episode(episode_dir, cameras, nhand, num_history, trajectory_length,
                    keyframe_gripper_change, keyframe_velocity_threshold,
                    keyframe_distance_threshold=None,
                    input_quat_format='wxyz'):
    """
    Process a single episode directory into arrays ready for zarr.

    Returns dict of arrays or None if episode is invalid.
    """
    ncam = len(cameras)

    # Load EEF states
    eef_path = os.path.join(episode_dir, "eef_states.npy")
    if not os.path.exists(eef_path):
        print(f"  Skipping {episode_dir}: missing eef_states.npy")
        return None
    eef_states = np.load(eef_path).astype(np.float32)  # (T, nhand*8)
    T = len(eef_states)

    if T < 2:
        print(f"  Skipping {episode_dir}: too few timesteps ({T})")
        return None

    # Convert quaternion format to xyzw if needed
    if not input_quat_format:
        raise ValueError("input_quat_format is required (wxyz or xyzw)")
    if input_quat_format == 'wxyz':
        eef_states = _convert_quat_wxyz_to_xyzw(eef_states)

    # Load camera parameters
    extrinsics_raw = np.load(
        os.path.join(episode_dir, "camera_extrinsics.npy")
    ).astype(np.float16)
    intrinsics = np.load(
        os.path.join(episode_dir, "camera_intrinsics.npy")
    ).astype(np.float16)  # (ncam, 3, 3)

    # Per-frame extrinsics: (T, ncam, 4, 4), static: (ncam, 4, 4)
    per_frame_ext = (extrinsics_raw.ndim == 4 and extrinsics_raw.shape[1] == ncam)
    if per_frame_ext:
        assert extrinsics_raw.shape == (T, ncam, 4, 4), \
            f"Expected extrinsics shape ({T}, {ncam}, 4, 4), got {extrinsics_raw.shape}"
    else:
        assert extrinsics_raw.shape == (ncam, 4, 4), \
            f"Expected extrinsics shape ({ncam}, 4, 4), got {extrinsics_raw.shape}"

    # Handle camera ordering from camera_names.json if present
    data_cam_names = _read_camera_names(episode_dir)
    cam_reorder_map = None
    if data_cam_names is not None and data_cam_names != cameras:
        if per_frame_ext:
            # Reorder per-frame: (T, ncam, 4, 4) -> reorder ncam axis
            data_cam_list = data_cam_names if isinstance(data_cam_names, list) else list(data_cam_names)
            cam_map = [data_cam_list.index(c) for c in cameras]
            extrinsics_raw = extrinsics_raw[:, cam_map]
            intrinsics = intrinsics[cam_map]
            cam_reorder_map = cam_map
        else:
            extrinsics_raw, intrinsics, cam_reorder_map = _reorder_cameras(
                data_cam_names, cameras, extrinsics_raw, intrinsics
            )

    assert intrinsics.shape == (ncam, 3, 3), \
        f"Expected intrinsics shape ({ncam}, 3, 3), got {intrinsics.shape}"

    # Detect keyframes or use all timesteps
    if keyframe_distance_threshold is not None:
        keyframes = detect_keyframes(
            eef_states, distance_threshold=keyframe_distance_threshold
        )
    elif keyframe_gripper_change:
        keyframes = detect_keyframes(
            eef_states, gripper_change=True,
            vel_threshold=keyframe_velocity_threshold
        )
    else:
        keyframes = list(range(T))

    if len(keyframes) < 2:
        print(f"  Skipping {episode_dir}: fewer than 2 keyframes")
        return None

    # Build proprioception: (num_samples, num_history, nhand, 8)
    # We use keyframes[:-1] as observation points, keyframes[1:] as action targets
    obs_keyframes = keyframes[:-1]
    prop = build_proprioception(eef_states, keyframes, num_history, nhand)
    prop = prop[:-1]  # drop the last (no action target for it)

    # Build actions: (num_samples, trajectory_length, nhand, 8)
    actions = build_actions(eef_states, keyframes, trajectory_length, nhand)
    num_samples = len(actions)

    # Determine image loading order: if camera_names.json differs from target,
    # load using the data's camera names but reorder to target order
    load_cam_names = data_cam_names if data_cam_names is not None else cameras

    # Load RGB and depth at observation keyframes
    rgbs = []
    depths = []
    for kf in obs_keyframes:
        rgb = load_rgb(episode_dir, load_cam_names, kf)
        dep = load_depth(episode_dir, load_cam_names, kf)
        if cam_reorder_map is not None:
            rgb = rgb[cam_reorder_map]
            dep = dep[cam_reorder_map]
        rgbs.append(rgb)
        depths.append(dep)
    rgbs = np.stack(rgbs)      # (num_samples, ncam, 3, H, W)
    depths = np.stack(depths)  # (num_samples, ncam, H, W)

    # Camera params per sample
    if per_frame_ext:
        # Index per-frame extrinsics at observation keyframes
        extr = np.stack([extrinsics_raw[kf] for kf in obs_keyframes])  # (num_samples, ncam, 4, 4)
    else:
        extr = np.tile(extrinsics_raw[None], (num_samples, 1, 1, 1))  # (num_samples, ncam, 4, 4)
    intr = np.tile(intrinsics[None], (num_samples, 1, 1, 1))  # (num_samples, ncam, 3, 3)

    result = {
        "rgb": rgbs,
        "depth": depths,
        "proprioception": prop,
        "action": actions,
        "extrinsics": extr,
        "intrinsics": intr,
    }

    # Optionally include joint states if available
    joint_path = os.path.join(episode_dir, "joint_states.npy")
    if os.path.exists(joint_path):
        joint_states = np.load(joint_path).astype(np.float32)
        # Build joint state samples at observation keyframes
        joint_samples = joint_states[obs_keyframes[:num_samples]]
        result["joint_states"] = joint_samples

    return result


def create_zarr(filename, ncam, nhand, num_history, trajectory_length,
                num_joints=None):
    """Initialize an empty zarr store with the correct schema."""
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
    zarr_file = zarr.open_group(filename, mode="w")

    def _create(field, shape, dtype):
        zarr_file.create_dataset(
            field,
            shape=(0,) + shape,
            chunks=(1,) + shape,
            compressor=compressor,
            dtype=dtype,
        )

    _create("rgb", (ncam, 3, IM_SIZE, IM_SIZE), "uint8")
    _create("depth", (ncam, IM_SIZE, IM_SIZE), "float16")
    _create("proprioception", (num_history, nhand, 8), "float32")
    _create("action", (trajectory_length, nhand, 8), "float32")
    _create("extrinsics", (ncam, 4, 4), "float16")
    _create("intrinsics", (ncam, 3, 3), "float16")
    _create("task_id", (), "uint8")
    _create("variation", (), "uint8")
    if num_joints is not None:
        _create("joint_states", (num_joints,), "float32")

    return zarr_file


def main():
    args = _resolve_args(parse_arguments())
    ncam = len(args.cameras)
    task2id = {task: i for i, task in enumerate(args.tasks)}

    # Collect all episodes with their task labels
    all_episodes = []  # list of (episode_dir, task_name, variation)
    for task in args.tasks:
        task_dir = os.path.join(args.root, task)
        if not os.path.isdir(task_dir):
            print(f"Warning: task directory {task_dir} not found, skipping")
            continue
        episodes = sorted([
            d for d in os.listdir(task_dir)
            if os.path.isdir(os.path.join(task_dir, d)) and d.startswith("episode")
        ])
        # Load per-task instructions for variation mapping
        instr_path = os.path.join(task_dir, "instructions.json")
        variations = {}
        if os.path.exists(instr_path):
            variations = json.load(open(instr_path))

        for ep in episodes:
            # Try to detect variation from a variation.txt or default to 0
            var_file = os.path.join(task_dir, ep, "variation.txt")
            if os.path.exists(var_file):
                with open(var_file) as f:
                    var = int(f.read().strip())
            else:
                var = 0
            all_episodes.append((os.path.join(task_dir, ep), task, var))

    if not all_episodes:
        print("No episodes found. Check --root and --tasks arguments.")
        return

    # Detect joint state dimensions from first episode
    num_joints = None
    first_ep_dir = all_episodes[0][0]
    joint_path = os.path.join(first_ep_dir, "joint_states.npy")
    if os.path.exists(joint_path):
        num_joints = np.load(joint_path).shape[1]
        print(f"Detected joint states with {num_joints} joints")

    if args.input_quat_format == 'wxyz':
        print(f"Quaternion input format: wxyz (will convert to xyzw)")
    else:
        print(f"Quaternion input format: xyzw (no conversion needed)")

    # Shuffle and split train/val
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_episodes))
    n_val = int(len(all_episodes) * args.val_ratio)
    # With very few episodes, use same data for both splits
    if n_val == 0 or len(all_episodes) - n_val == 0:
        train_indices = set(indices)
        val_indices = set(indices)
    else:
        val_indices = set(indices[:n_val])
        train_indices = set(indices[n_val:])

    print(f"Found {len(all_episodes)} episodes: "
          f"{len(train_indices)} train, {len(val_indices)} val")

    # Create output directory
    os.makedirs(args.tgt, exist_ok=True)

    # Save train/val split info
    split_info = {
        "seed": 42,
        "val_ratio": args.val_ratio,
        "train_episodes": [
            os.path.basename(all_episodes[i][0]) for i in sorted(train_indices)
        ],
        "val_episodes": [
            os.path.basename(all_episodes[i][0]) for i in sorted(val_indices)
        ],
    }
    split_info_path = os.path.join(args.tgt, "split_info.json")
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Split info saved to {split_info_path}")

    # Process train and val splits
    for split, split_indices in [("train", train_indices), ("val", val_indices)]:
        zarr_path = os.path.join(args.tgt, f"{split}.zarr")
        if os.path.exists(zarr_path):
            print(f"{zarr_path} already exists, skipping. Delete it to regenerate.")
            continue

        zarr_file = create_zarr(
            zarr_path, ncam, args.nhand, args.num_history, args.trajectory_length,
            num_joints=num_joints
        )
        total_samples = 0

        for i in tqdm(sorted(split_indices), desc=f"Processing {split}"):
            episode_dir, task, variation = all_episodes[i]

            data = process_episode(
                episode_dir, args.cameras, args.nhand, args.num_history,
                args.trajectory_length,
                args.keyframe_gripper_change, args.keyframe_velocity_threshold,
                keyframe_distance_threshold=args.keyframe_distance_threshold,
                input_quat_format=args.input_quat_format
            )
            if data is None:
                continue

            num_samples = len(data["action"])
            task_ids = np.full(num_samples, task2id[task], dtype=np.uint8)
            variations = np.full(num_samples, variation, dtype=np.uint8)

            zarr_file["rgb"].append(data["rgb"])
            zarr_file["depth"].append(data["depth"])
            zarr_file["proprioception"].append(data["proprioception"])
            zarr_file["action"].append(data["action"])
            zarr_file["extrinsics"].append(data["extrinsics"])
            zarr_file["intrinsics"].append(data["intrinsics"])
            zarr_file["task_id"].append(task_ids)
            zarr_file["variation"].append(variations)
            if "joint_states" in data and "joint_states" in zarr_file:
                zarr_file["joint_states"].append(data["joint_states"])

            total_samples += num_samples

        print(f"  {split}: {total_samples} samples written to {zarr_path}")

    # Build instructions JSON
    instructions = {}
    for task in args.tasks:
        task_dir = os.path.join(args.root, task)
        instr_path = os.path.join(task_dir, "instructions.json")
        if os.path.exists(instr_path):
            instructions[task] = json.load(open(instr_path))
        else:
            # Default instruction from task name
            default_instr = task.replace("_", " ")
            instructions[task] = {"0": [default_instr]}
            print(f"  No instructions.json for {task}, using default: '{default_instr}'")

    instr_out_dir = os.path.join("instructions", "xarm")
    os.makedirs(instr_out_dir, exist_ok=True)
    instr_out_path = os.path.join(instr_out_dir, "instructions.json")
    with open(instr_out_path, "w") as f:
        json.dump(instructions, f, indent=2)
    print(f"Instructions saved to {instr_out_path}")

    # Print summary for training script
    print("\n=== Ready for training ===")
    print(f"  --train_data_dir {os.path.join(args.tgt, 'train.zarr')}")
    print(f"  --eval_data_dir {os.path.join(args.tgt, 'val.zarr')}")
    print(f"  --train_instructions {instr_out_path}")
    print(f"  --val_instructions {instr_out_path}")
    print(f"  --bimanual {'true' if args.nhand == 2 else 'false'}")
    print(f"  --keypose_only {'true' if args.trajectory_length == 1 else 'false'}")
    print(f"  --num_history {args.num_history}")


if __name__ == "__main__":
    main()
