"""
Convert real robot arm demonstration data to Zarr format for 3D FlowMatch Actor training.

Expected input directory structure:
    <root>/
        <task_name>/
            episode_0/
                rgb/
                    <cam_name>_0000.png   # 256x256 RGB images
                    <cam_name>_0001.png
                    ...
                depth/
                    <cam_name>_0000.png   # 16-bit PNG depth (millimeters)
                    <cam_name>_0001.png
                    ...
                eef_states.npy            # (T, 8) float32: [x,y,z, qx,qy,qz,qw, gripper_open]
                camera_extrinsics.npy     # (ncam, 4, 4) float32: cam-to-world transforms
                camera_intrinsics.npy     # (ncam, 3, 3) float32: intrinsic matrices
            episode_1/
                ...
            instructions.json             # {"0": ["pick up the red cup", ...]}

Output: train.zarr/ and val.zarr/ compatible with the existing dataset classes.

Usage:
    python -m data_processing.real_arm_to_zarr \
        --root /path/to/demos \
        --tgt /path/to/output \
        --cameras front wrist \
        --nhand 1 \
        --tasks pick_cup place_cup \
        --val_ratio 0.1 \
        --trajectory_length 1 \
        --depth_scale 1000.0
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert real arm demos to zarr for 3D FlowMatch Actor"
    )
    parser.add_argument(
        "--root", type=str, required=True,
        help="Root directory containing task folders with episodes"
    )
    parser.add_argument(
        "--tgt", type=str, required=True,
        help="Output directory for zarr files"
    )
    parser.add_argument(
        "--cameras", type=str, nargs="+", default=["front", "wrist"],
        help="Camera names, matching filenames in rgb/ and depth/ folders"
    )
    parser.add_argument(
        "--nhand", type=int, default=1, choices=[1, 2],
        help="Number of arms (1=single, 2=bimanual)"
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", required=True,
        help="List of task folder names"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Fraction of episodes to use for validation"
    )
    parser.add_argument(
        "--trajectory_length", type=int, default=1,
        help="Action trajectory length. 1=keypose only, >1=interpolated trajectory"
    )
    parser.add_argument(
        "--depth_scale", type=float, default=1000.0,
        help="Divisor to convert raw depth to meters (1000 for mm, 1 if already meters)"
    )
    parser.add_argument(
        "--num_history", type=int, default=3,
        help="Number of proprioception history steps"
    )
    parser.add_argument(
        "--keyframe_gripper_change", action="store_true",
        help="Auto-detect keyframes from gripper state changes instead of using all timesteps"
    )
    parser.add_argument(
        "--keyframe_velocity_threshold", type=float, default=0.01,
        help="Velocity threshold for keyframe detection (m/s)"
    )
    return parser.parse_args()


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


def load_depth(episode_dir, cameras, timestep, depth_scale):
    """Load depth maps for all cameras at a given timestep. Returns (ncam, H, W) float16."""
    depths = []
    for cam in cameras:
        path = os.path.join(episode_dir, "depth", f"{cam}_{timestep:04d}.png")
        raw = np.array(Image.open(path))
        # Convert to meters
        depth_m = raw.astype(np.float32) / depth_scale
        # Resize if needed
        if depth_m.shape != (IM_SIZE, IM_SIZE):
            depth_m = np.array(
                Image.fromarray(depth_m).resize((IM_SIZE, IM_SIZE), Image.NEAREST)
            )
        depths.append(depth_m.astype(np.float16))
    return np.stack(depths)  # (ncam, H, W)


def detect_keyframes(eef_states, gripper_change=True, vel_threshold=0.01):
    """
    Detect keyframes from EEF trajectory.

    Keyframes are detected at:
    - Gripper state changes (open -> close or close -> open)
    - Low-velocity stops (when the arm pauses)
    - First and last timestep

    Args:
        eef_states: (T, nhand*8) array of EEF states
        gripper_change: Whether to use gripper changes for detection
        vel_threshold: Velocity threshold for stop detection

    Returns:
        List of keyframe indices (always includes 0 and T-1)
    """
    T = len(eef_states)
    if T <= 2:
        return list(range(T))

    keyframes = set([0, T - 1])

    # Per-arm analysis (handles both single and bimanual)
    arm_dim = 8
    n_arms = eef_states.shape[1] // arm_dim if eef_states.ndim == 1 else 1
    if eef_states.ndim == 1:
        eef_states = eef_states.reshape(T, -1)

    for arm in range(max(1, eef_states.shape[1] // arm_dim)):
        start = arm * arm_dim
        positions = eef_states[:, start:start + 3]
        gripper = eef_states[:, start + 7]

        # Gripper state changes
        if gripper_change:
            gripper_binary = (gripper > 0.5).astype(int)
            changes = np.where(np.diff(gripper_binary) != 0)[0] + 1
            keyframes.update(changes.tolist())

        # Velocity-based stops (position only)
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        stopped = velocities < vel_threshold
        # Find transitions from moving to stopped
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


def process_episode(episode_dir, cameras, nhand, num_history, trajectory_length,
                    depth_scale, keyframe_gripper_change, keyframe_velocity_threshold):
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

    # Load camera parameters (constant across episode)
    extrinsics = np.load(
        os.path.join(episode_dir, "camera_extrinsics.npy")
    ).astype(np.float16)  # (ncam, 4, 4)
    intrinsics = np.load(
        os.path.join(episode_dir, "camera_intrinsics.npy")
    ).astype(np.float16)  # (ncam, 3, 3)

    assert extrinsics.shape == (ncam, 4, 4), \
        f"Expected extrinsics shape ({ncam}, 4, 4), got {extrinsics.shape}"
    assert intrinsics.shape == (ncam, 3, 3), \
        f"Expected intrinsics shape ({ncam}, 3, 3), got {intrinsics.shape}"

    # Detect keyframes or use all timesteps
    if keyframe_gripper_change:
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

    # Load RGB and depth at observation keyframes
    rgbs = []
    depths = []
    for kf in obs_keyframes:
        rgbs.append(load_rgb(episode_dir, cameras, kf))
        depths.append(load_depth(episode_dir, cameras, kf, depth_scale))
    rgbs = np.stack(rgbs)      # (num_samples, ncam, 3, H, W)
    depths = np.stack(depths)  # (num_samples, ncam, H, W)

    # Replicate camera params for each sample
    extr = np.tile(extrinsics[None], (num_samples, 1, 1, 1))  # (num_samples, ncam, 4, 4)
    intr = np.tile(intrinsics[None], (num_samples, 1, 1, 1))  # (num_samples, ncam, 3, 3)

    return {
        "rgb": rgbs,
        "depth": depths,
        "proprioception": prop,
        "action": actions,
        "extrinsics": extr,
        "intrinsics": intr,
    }


def create_zarr(filename, ncam, nhand, num_history, trajectory_length):
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

    return zarr_file


def main():
    args = parse_arguments()
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

    # Shuffle and split train/val
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_episodes))
    n_val = max(1, int(len(all_episodes) * args.val_ratio))
    val_indices = set(indices[:n_val])
    train_indices = set(indices[n_val:])

    print(f"Found {len(all_episodes)} episodes: "
          f"{len(train_indices)} train, {len(val_indices)} val")

    # Create output directory
    os.makedirs(args.tgt, exist_ok=True)

    # Process train and val splits
    for split, split_indices in [("train", train_indices), ("val", val_indices)]:
        zarr_path = os.path.join(args.tgt, f"{split}.zarr")
        if os.path.exists(zarr_path):
            print(f"{zarr_path} already exists, skipping. Delete it to regenerate.")
            continue

        zarr_file = create_zarr(
            zarr_path, ncam, args.nhand, args.num_history, args.trajectory_length
        )
        total_samples = 0

        for i in tqdm(sorted(split_indices), desc=f"Processing {split}"):
            episode_dir, task, variation = all_episodes[i]

            data = process_episode(
                episode_dir, args.cameras, args.nhand, args.num_history,
                args.trajectory_length, args.depth_scale,
                args.keyframe_gripper_change, args.keyframe_velocity_threshold
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

    instr_out_dir = os.path.join("instructions", "real_arm")
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
