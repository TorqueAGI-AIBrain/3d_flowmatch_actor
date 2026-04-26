# verify_tf_batch.py: Batch verify TF corrections across all episodes in a bag directory.
# verify_tf_batch.py: Reports per-episode table-Z agreement between front and wrist cameras.

"""
For each episode bag, pick a mid-episode frame, unproject front and wrist
depth to link_base using the corrected /front/pose and /wrist/pose topics,
and report alignment metrics:
  - Table surface Z peak for each camera (should match)
  - Z offset between cameras (should be ~0)
  - Point count per camera

Usage:
    python -m data_processing.verify_tf_batch
    python -m data_processing.verify_tf_batch --bag_dir data/xarm/default_task_fixed_v2
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from data_processing.tf_fixes import (
    TYPESTORE,
    compute_world_T_wrist,
    decode_depth,
    decode_K,
    decode_rgb,
    decode_pose_mat,
    read_frame,
    save_ply,
    transform_points,
    unproject_to_camera_frame,
)


def find_table_z(pts_base, z_min=-0.05, z_max=0.15, n_bins=100):
    """Find the dominant Z value (table surface) via histogram peak."""
    z = pts_base[:, 2]
    mask = (z > z_min) & (z < z_max)
    if mask.sum() < 100:
        return np.nan, 0
    z_table = z[mask]
    counts, edges = np.histogram(z_table, bins=n_bins)
    peak_idx = counts.argmax()
    peak_z = (edges[peak_idx] + edges[peak_idx + 1]) / 2
    return peak_z, int(mask.sum())


def verify_episode(mcap_path, frame_idx=None):
    """Verify one episode. Returns dict with metrics."""
    collected, tf_static, tf_dynamic, total = read_frame(mcap_path, frame_idx=0)

    # Pick mid-episode frame if not specified
    if frame_idx is None:
        frame_idx = total // 2
    if frame_idx >= total:
        frame_idx = total - 1

    collected, tf_static, tf_dynamic, total = read_frame(mcap_path, frame_idx=frame_idx)

    # Front camera
    _, front_depth_msg = collected['/front/depth/image_raw']
    _, front_info_msg = collected['/front/camera_info']
    _, front_pose_msg = collected['/front/pose']

    front_depth = decode_depth(front_depth_msg)
    front_K = decode_K(front_info_msg)
    world_T_front = decode_pose_mat(front_pose_msg)

    front_pts_cam, front_valid = unproject_to_camera_frame(front_depth, front_K)
    front_pts_base = transform_points(front_pts_cam, world_T_front)

    # Wrist camera
    _, wrist_depth_msg = collected['/wrist/depth/image_raw']
    _, wrist_info_msg = collected['/wrist/camera_info']
    _, wrist_pose_msg = collected['/wrist/pose']

    wrist_depth = decode_depth(wrist_depth_msg)
    wrist_K = decode_K(wrist_info_msg)
    world_T_wrist = decode_pose_mat(wrist_pose_msg)

    wrist_pts_cam, wrist_valid = unproject_to_camera_frame(wrist_depth, wrist_K)
    wrist_pts_base = transform_points(wrist_pts_cam, world_T_wrist)

    # Table Z analysis
    front_table_z, front_table_pts = find_table_z(front_pts_base)
    wrist_table_z, wrist_table_pts = find_table_z(wrist_pts_base)
    z_offset = wrist_table_z - front_table_z if not (np.isnan(front_table_z) or np.isnan(wrist_table_z)) else np.nan

    return {
        'total_frames': total,
        'frame_idx': frame_idx,
        'front_pts': len(front_pts_cam),
        'wrist_pts': len(wrist_pts_cam),
        'front_table_z': front_table_z,
        'wrist_table_z': wrist_table_z,
        'z_offset_cm': z_offset * 100 if not np.isnan(z_offset) else np.nan,
        'front_table_pts': front_table_pts,
        'wrist_table_pts': wrist_table_pts,
        'front_z_range': (float(front_pts_base[:, 2].min()), float(front_pts_base[:, 2].max())),
        'wrist_z_range': (float(wrist_pts_base[:, 2].min()), float(wrist_pts_base[:, 2].max())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', type=str, default='data/xarm/default_task_fixed_v2')
    parser.add_argument('--save_ply', action='store_true',
                        help='Also save PLY files for each episode')
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    episode_dirs = sorted([d for d in bag_dir.iterdir()
                           if d.is_dir() and d.name.startswith('episode_')])

    print(f"Verifying {len(episode_dirs)} episodes in {bag_dir}")
    print(f"{'Episode':>20s} {'Frame':>6s} {'FrontPts':>9s} {'WristPts':>9s} "
          f"{'FrontZ':>7s} {'WristZ':>7s} {'Offset':>8s} {'Status':>8s}")
    print("-" * 85)

    results = []
    for ep_dir in episode_dirs:
        mcap_files = list(ep_dir.glob('*.mcap'))
        if not mcap_files:
            continue

        ep_name = ep_dir.name.replace('_bag', '')
        try:
            r = verify_episode(mcap_files[0])
            status = 'OK' if abs(r['z_offset_cm']) < 3.0 else 'WARN' if abs(r['z_offset_cm']) < 5.0 else 'BAD'
            if np.isnan(r['z_offset_cm']):
                status = 'NO_TBL'

            print(f"{ep_name:>20s} {r['frame_idx']:>6d} {r['front_pts']:>9d} {r['wrist_pts']:>9d} "
                  f"{r['front_table_z']:>7.3f} {r['wrist_table_z']:>7.3f} "
                  f"{r['z_offset_cm']:>7.2f}cm {status:>8s}")
            results.append(r)
        except Exception as e:
            print(f"{ep_name:>20s} {'ERROR':>6s}  {str(e)[:60]}")

    # Summary
    offsets = [r['z_offset_cm'] for r in results if not np.isnan(r['z_offset_cm'])]
    if offsets:
        print(f"\n{'=' * 85}")
        print(f"Summary ({len(offsets)} episodes with table surface detected):")
        print(f"  Mean Z offset: {np.mean(offsets):.2f} cm")
        print(f"  Std Z offset:  {np.std(offsets):.2f} cm")
        print(f"  Max |offset|:  {max(abs(o) for o in offsets):.2f} cm")
        print(f"  Episodes within 3cm: {sum(1 for o in offsets if abs(o) < 3.0)}/{len(offsets)}")


if __name__ == '__main__':
    main()
