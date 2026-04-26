# calibrate_front.py: Verify front+wrist calibration corrections at any frame in the bag.
# calibrate_front.py: Outputs corrected PLYs with coordinate axes for 3D viewer verification.

"""
Apply front and wrist calibration corrections and export PLYs for verification.
Use --frame to pick different time points and confirm corrections hold across
different arm configurations.

Usage:
    python -m data_processing.calibrate_front              # first frame
    python -m data_processing.calibrate_front --frame 50   # 50th frame
    python -m data_processing.calibrate_front --frame -1   # last frame

Outputs:
    data_processing/debug_pcds/front_calibrated.ply   — front PCD in corrected link_base
    data_processing/debug_pcds/origin_axes.ply        — RGB axes at link_base origin
    data_processing/debug_pcds/wrist_calibrated.ply   — wrist PCD in corrected link_base
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from data_processing.tf_fixes import (
    MCAP_FILE,
    OUTPUT_DIR,
    compute_world_T_wrist,
    decode_depth,
    decode_K,
    decode_rgb,
    make_mat,
    read_frame,
    save_ply,
    transform_points,
    unproject_to_camera_frame,
)

# =====================================================================
# MANUAL TRANSFORM: link_base -> front_optical_frame
# Edit these values and re-run to iterate on calibration.
# Obtained by aligning origin_axes.ply to robot base plate in CloudCompare.
# =====================================================================

# Front correction: origin_axes.ply was moved to this pose in CloudCompare
# to align with the robot base plate visible in the front PCD.
# Applied as: corrected_front = inv(FRONT_CORRECTION) @ /front/pose @ pts_cam
# fmt: off
FRONT_CORRECTION = np.array([
    [0.990103,  0.139081, -0.018772, 0.036359],
    [-0.133884, 0.976160,  0.170840, 0.000056],
    [0.042085, -0.166636,  0.985120, 0.067018],
    [0.000000,  0.000000,  0.000000, 1.000000],
], dtype=np.float32)
# fmt: on

# Wrist correction: wrist PCD was shifted by this transform in CloudCompare
# to align with the corrected front PCD. Pure translation (13cm Z + minor XY).
# Applied as: corrected_wrist = WRIST_CORRECTION @ TF_chain @ pts_cam
# fmt: off
WRIST_CORRECTION = np.array([
    [1.000000, 0.000000, 0.000000, -0.003120],
    [0.000000, 1.000000, 0.000000, -0.009791],
    [0.000000, 0.000000, 1.000000,  0.130149],
    [0.000000, 0.000000, 0.000000,  1.000000],
], dtype=np.float32)
# fmt: on


# =====================================================================


def build_axes_ply(length=0.15, n_pts_per_axis=50):
    """Create colored point clouds for X (red), Y (green), Z (blue) axes."""
    pts_list = []
    col_list = []
    for axis_idx, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        for t in np.linspace(0, length, n_pts_per_axis):
            pt = [0.0, 0.0, 0.0]
            pt[axis_idx] = t
            pts_list.append(pt)
            col_list.append(color)
    return np.array(pts_list, np.float32), np.array(col_list, np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to extract (0=first, -1=last)')
    args = parser.parse_args()

    frame_idx = args.frame
    print(f"Reading {MCAP_FILE}")
    collected, tf_static, tf_dynamic, total = read_frame(
        MCAP_FILE, frame_idx=0 if frame_idx >= 0 else 0)

    # Re-read with actual index (handle negative indexing)
    if frame_idx < 0:
        frame_idx = total + frame_idx
    if frame_idx != 0:
        collected, tf_static, tf_dynamic, total = read_frame(MCAP_FILE, frame_idx=frame_idx)

    print(f"Frame {frame_idx} / {total} total")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Front: inv(FRONT_CORRECTION) @ /front/pose ---
    from data_processing.tf_fixes import decode_pose_mat as _decode_pose
    _, front_pose_msg = collected['/front/pose']
    original_front_T = _decode_pose(front_pose_msg)
    world_T_front = (np.linalg.inv(FRONT_CORRECTION) @ original_front_T).astype(np.float32)

    t = world_T_front[:3, 3]
    euler = Rotation.from_matrix(world_T_front[:3, :3]).as_euler('xyz', degrees=True)
    quat_xyzw = Rotation.from_matrix(world_T_front[:3, :3]).as_quat()
    print(f"\n=== Front: corrected link_base -> front_optical_frame ===")
    print(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"  Euler (xyz deg): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]")
    print(f"  Quaternion (xyzw): [{quat_xyzw[0]:.6f}, {quat_xyzw[1]:.6f}, {quat_xyzw[2]:.6f}, {quat_xyzw[3]:.6f}]")

    # --- Wrist: WRIST_CORRECTION @ TF_chain ---
    wrist_t_stamp = collected['/wrist/color/image_raw'][0]
    world_T_wrist_raw = compute_world_T_wrist(tf_static, tf_dynamic, wrist_t_stamp)
    world_T_wrist = (WRIST_CORRECTION @ world_T_wrist_raw).astype(np.float32)

    wt = world_T_wrist[:3, 3]
    weuler = Rotation.from_matrix(world_T_wrist[:3, :3]).as_euler('xyz', degrees=True)
    print(f"\n=== Wrist: corrected link_base -> wrist_optical_frame ===")
    print(f"  Translation: [{wt[0]:.4f}, {wt[1]:.4f}, {wt[2]:.4f}]")
    print(f"  Euler (xyz deg): [{weuler[0]:.1f}, {weuler[1]:.1f}, {weuler[2]:.1f}]")
    print(f"  Wrist correction (translation): [{WRIST_CORRECTION[0,3]:.4f}, {WRIST_CORRECTION[1,3]:.4f}, {WRIST_CORRECTION[2,3]:.4f}]")

    # --- Front camera ---
    _, front_rgb_msg = collected['/front/color/image_raw']
    _, front_depth_msg = collected['/front/depth/image_raw']
    _, front_info_msg = collected['/front/camera_info']

    front_rgb = decode_rgb(front_rgb_msg)
    front_depth = decode_depth(front_depth_msg)
    front_K = decode_K(front_info_msg)

    front_pts_cam, front_valid = unproject_to_camera_frame(front_depth, front_K)
    front_colors = front_rgb.reshape(-1, 3)[front_valid]

    front_pts_base = transform_points(front_pts_cam, world_T_front)
    print(f"\n  Front PCD: {len(front_pts_base)} points")
    print(f"  Z range: [{front_pts_base[:,2].min():.3f}, {front_pts_base[:,2].max():.3f}]")
    print(f"  X range: [{front_pts_base[:,0].min():.3f}, {front_pts_base[:,0].max():.3f}]")
    print(f"  Y range: [{front_pts_base[:,1].min():.3f}, {front_pts_base[:,1].max():.3f}]")

    # Output to per-frame subdirectory
    frame_dir = OUTPUT_DIR / f'frame_{frame_idx:03d}'
    frame_dir.mkdir(parents=True, exist_ok=True)

    save_ply(frame_dir / 'front_calibrated.ply', front_pts_base, front_colors)
    print(f"  Saved: {frame_dir}/front_calibrated.ply")

    # --- Coordinate frame axes at link_base origin ---
    axes_pts, axes_colors = build_axes_ply(length=0.15)
    save_ply(frame_dir / 'origin_axes.ply', axes_pts, axes_colors)
    print(f"  Saved: {frame_dir}/origin_axes.ply (R=X, G=Y, B=Z, length=15cm)")

    # --- Wrist camera (corrected TF chain) ---
    _, wrist_rgb_msg = collected['/wrist/color/image_raw']
    _, wrist_depth_msg = collected['/wrist/depth/image_raw']
    _, wrist_info_msg = collected['/wrist/camera_info']

    wrist_rgb = decode_rgb(wrist_rgb_msg)
    wrist_depth = decode_depth(wrist_depth_msg)
    wrist_K = decode_K(wrist_info_msg)

    wrist_pts_cam, wrist_valid = unproject_to_camera_frame(wrist_depth, wrist_K)
    wrist_colors = wrist_rgb.reshape(-1, 3)[wrist_valid]

    wrist_pts_base = transform_points(wrist_pts_cam, world_T_wrist)
    print(f"\n  Wrist PCD: {len(wrist_pts_base)} points (corrected TF chain)")
    print(f"  Z range: [{wrist_pts_base[:,2].min():.3f}, {wrist_pts_base[:,2].max():.3f}]")

    save_ply(frame_dir / 'wrist_calibrated.ply', wrist_pts_base, wrist_colors)
    print(f"  Saved: {frame_dir}/wrist_calibrated.ply")

    # --- Summary ---
    print(f"\n=== Output: {frame_dir}/ ===")
    print(f"  origin_axes.ply, front_calibrated.ply, wrist_calibrated.ply")


if __name__ == '__main__':
    main()
