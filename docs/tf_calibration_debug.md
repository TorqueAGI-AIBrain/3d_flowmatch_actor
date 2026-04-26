# TF Calibration Debug: Wrist-Front Camera Point Cloud Misalignment

## Problem

When transforming RGBD point clouds from the front and wrist cameras to `link_base` using their respective `/front/pose` and `/wrist/pose` topics, the resulting world-frame point clouds do not align properly. The wrist camera cloud appears shifted lower than expected relative to the front camera cloud.

## Data Source

- **Bag files**: `data/xarm/default_task/episode_*_bag/` (rosbag2 v9, Jazzy, MCAP format)
- **Robot**: xArm6 with wrist-mounted RealSense D435i and front-mounted Azure Kinect
- **36 episodes** of `place_wrench` task

## TF Tree (from old bags)

The `/tf` topic publishes the kinematic chain:

```
link_base → link1 → link2 → link3 → link4 → link5 → link6
```

**Disconnected subtree** (not connected to link6):
```
xarm_gripper_base_link → left_outer_knuckle → left_finger
                       → left_inner_knuckle
                       → right_outer_knuckle → right_finger
                       → right_inner_knuckle
```

**Missing links in the chain**:
```
link6 → ft_sensor_link → link_eef → xarm_gripper_base_link
```

The physical arm has a force-torque sensor between link6 and the EEF flange. The wrist camera is mounted somewhere in this region. But these intermediate frames are not published in `/tf` or `/tf_static`.

**No camera frames exist in TF** — no `camera_wrist`, `camera_wrist_optical_frame`, or similar.

## Pose Topics

All three pose topics publish in `link_base` frame:

| Topic | Position (first frame, ep0) | Description |
|-------|----------------------------|-------------|
| `/front/pose` | (0.606, 0.493, 0.828) | Front camera (Azure Kinect), static |
| `/wrist/pose` | (0.346, -0.072, 0.211) | Wrist camera (RealSense D435i), dynamic (eye-in-hand) |
| `/eef_pose` | (0.272, -0.083, 0.323) | End-effector flange |

## Computed Transforms

From the TF chain at the first frame of episode 0:

| Frame | World Position | Notes |
|-------|---------------|-------|
| `link6` | (0.270, -0.083, 0.414) | From TF chain |
| `link_eef` (computed) | (0.272, -0.083, 0.323) | = `/eef_pose`, 9.1cm below link6 |
| `/wrist/pose` | (0.346, -0.072, 0.211) | 13.5cm from EEF, 20.3cm below link6 |

### link6_T_eef (from data)
```
Translation: [0.0, 0.0, -0.091]  (pure Z offset, 9.1cm below link6)
Rotation: identity
```

### eef_T_wrist_cam (from data)
```
Translation: [0.065, -0.033, 0.113]
Rotation (euler deg): [-0.15, -0.41, 87.92]
```

### Expected eef_T_cam (from torqueagi-arm PR #41 config)
```
Translation: [0.063, -0.033, 0.0194]
Quaternion (xyzw): [-0.0003797, -0.004984, 0.6933, 0.7206]
```

### Discrepancy
- X, Y match well: `0.065 vs 0.063`, `-0.033 vs -0.033`
- **Z is way off**: `0.113` (from data) vs `0.019` (from PR #41 config)
- Delta: **9.4cm in Z**

This Z discrepancy is approximately equal to `link6_T_eef.z = 0.091m`, suggesting `/wrist/pose` is being computed relative to `link6` (or some frame near it) rather than relative to the actual camera mount point.

### link6_T_wrist_cam (what /wrist/pose node computes)
```
Translation: [0.065, -0.033, 0.204]
```
This Z=0.204m places the camera 20.4cm below link6, which is too far. The correct value should be approximately:
```
link6_T_eef.z + eef_T_cam.z = 0.091 + 0.019 = 0.110m
```

## Point Cloud Analysis

### Camera specs
- **Front** (Azure Kinect): 1280x720, depth 32FC1 (metres), range 0.56-3.15m
- **Wrist** (RealSense D435i): 640x480, depth 32FC1 (metres), range 0.16-1.90m

### Unprojection results (manual, bypassing depth2cloud)

Using `world_T_cam @ K_inv @ [u*d, v*d, d, 1]`:

| Camera | Table Z peak | Points |
|--------|-------------|--------|
| Front | 0.043m | 469,124 |
| Wrist | 0.021m | 178,796 |
| **Offset** | **2.2cm** | |

The table surface should be at the same Z in world frame for both cameras. The 2.2cm offset indicates calibration error.

### Applying PR #41 correction (eef_T_cam)

Computing `world_T_cam = world_T_eef @ eef_T_cam`:
- Table Z from wrist shifted to **0.117m** (overshot, worse than uncorrected)
- This confirms the PR #41 `eef_T_cam` is for a different configuration

## Intrinsics Bug (Fixed)

A separate bug was found and fixed: images are resized to 256x256 but the camera intrinsic matrix K was stored at original resolution. This caused incorrect 3D unprojection during training. Fix applied in PR #4 (`fix/intrinsics-scaling`).

### Verified correct intrinsics (after fix)
| Camera | Original | Scaled to 256x256 |
|--------|----------|--------------------|
| Front | fx=600, cx=640 | fx=120, cx=128 |
| Wrist | fx=607, cx=308 | fx=243, cx=123 |

## Depth Convention

- Source bags: 32FC1 (float32, metres)
- Episode PNGs: uint16 (millimetres), `DEPTH_MM_SCALE = 1000.0`
- Zarr: float16 (metres)
- `depth_scale` config param removed; hardcoded as `DEPTH_MM_SCALE` constant

## Root Cause Hypothesis

The `/wrist/pose` topic is computed by a recording node that applies a `link6_T_cam` (or similar) transform, but:

1. The `link6 → ft_sensor_link → link_eef` chain is **not published in TF**
2. The gripper subtree (`xarm_gripper_base_link → ...`) is **disconnected from link6** in the TF tree
3. The camera mount offset used during recording appears to be wrong (Z=0.204 vs expected Z=0.110)

The fix needs to happen in the **robot arm recording software** — specifically:
1. Publish the complete TF chain: `link6 → ft_sensor_link → link_eef → xarm_gripper_base_link`
2. Add a static transform: `link_eef → camera_wrist_optical_frame` (or whichever link the camera is mounted to)
3. Compute `/wrist/pose` by looking up the camera frame in the full TF tree, not by applying a hardcoded offset

## New Bag Status

Episode 38 and 39 bags on external drive are **corrupted** (missing MCAP footer — recording was not finalized). Cannot use them to verify TF fixes. Need to re-record with the corrected TF tree.

## Visualization Files

Generated during debugging (in `offline_evaluation/results/`):
- `pcd_comparison.html` — buggy vs fixed intrinsics
- `pcd_rgb.html` — RGB-colored point clouds, both cameras
- `pcd_fullres_debug.html` — full-res PCD with camera positions
- `pcd_wrist_debug.html` — wrist PCD at 3 time points
- `pcd_wrist_calibration.html` — current vs PR#41-corrected wrist
- `pcd_z_correction.html` — table Z alignment attempt
- `pcd_manual_unproject.html` — manual unprojection verification
- `pcd_before_after.html` — zarr PCD before/after intrinsics fix
- `pcd_perframe_ext.html` — per-frame wrist extrinsics
- `pcd_zarr_fixed.html` — final zarr PCD

## Fixed Bag Analysis (default_task_fixed/episode_0_bag)

The TF tree was fixed to include the missing links. New static transforms added:

```
link6 → link_ft_adapter          t=(0, 0, 0)           identity
link_ft_adapter → link_ft_sensor t=(0, 0, 0.091)       FT sensor length
link_ft_sensor → link_eef        t=(0, 0, 0)           identity
link_eef → wrist_optical_frame   t=(0.063, -0.033, 0.119)  q=(-0.000380, -0.004985, 0.693304, 0.720628)
link_eef → xarm_gripper_base_link t=(0, 0, 0)          identity
link_base → front_optical_frame  t=(0.513, 0.375, 0.482)  q=(0.199, 0.857, -0.460, -0.117)
world → link_base                t=(0, 0, 0)           identity
xarm_gripper_base_link → link_tcp t=(0, 0, 0.172)
```

### Verification: time-synced TF-only point cloud alignment

Both point clouds were transformed to world frame using **only TF data** (no pose topics):
- Front: static `link_base → front_optical_frame` (verified correct)
- Wrist: dynamic chain `link_base → link1 → ... → link6 → link_ft_adapter → link_ft_sensor → link_eef → wrist_optical_frame` with per-frame joint angles time-synced to the depth image timestamp

**Result: still misaligned.** The front camera cloud is correct, but the wrist cloud does not overlap with the front cloud's view of the same scene.

### Root cause

The static transform `link_eef → wrist_optical_frame` is incorrect:
```
Current:  t=(0.063, -0.033, 0.119)  euler=(-0.4°, -0.4°, 87.8°)
```

This transform needs to be re-measured/re-calibrated on the physical robot. The translation and/or rotation of the wrist camera relative to the EEF flange is wrong.

### Visualization files
- `pcd_world_frame.html` — front (static TF) + wrist (dynamic TF chain, time-synced), both in world frame
- `pcd_tf_timesynced.html` — same but earlier iteration
- `pcd_4_variants.html` — comparison of pose topics vs TF chain
- `pcd_tf_only.html` — first attempt (not time-synced, incorrect)

## Key Findings

### 1. `link_base → front_optical_frame` TF static is WRONG

The TF static gives table Z = -0.30m (should be ~0). The `/front/pose` topic gives correct geometry (table Z ≈ 0.04m). These are completely different transforms:

| | TF static | /front/pose (correct) |
|---|---|---|
| Translation | (0.513, 0.375, 0.482) | (0.606, 0.493, 0.827) |
| Euler | (-123°, -1°, 153°) | (-124°, 8°, 130°) |

**Fix**: update `link_base → front_optical_frame` in the URDF/launch to match `/front/pose`.

### 2. `link_eef → wrist_optical_frame` Z is wrong

Original Z = 0.119m (11.9cm) is too large. Setting Z = 0.00119m (0.119cm) brings the wrist cloud to approximately the correct height and produces rough alignment with the front cloud.

Corrected transform:
```yaml
link_eef → wrist_optical_frame:
  translation: [0.063, -0.033, 0.00119]  # Z was 0.119, should be ~0.00119
  quaternion_xyzw: [-0.000380, -0.004985, 0.693304, 0.720628]
```

Remaining X/Y/rotation misalignment needs physical measurement or calibration target refinement.

### 3. `/front/pose` and `/wrist/pose` topics are more accurate than TF

The pose topics were computed with a different (better) calibration than the TF static transforms. For training data extraction, use the pose topics until the TF statics are corrected.

## Action Items

1. **Fix `link_base → front_optical_frame`** in URDF/launch — use the `/front/pose` values
2. **Fix `link_eef → wrist_optical_frame` Z** — change from 0.119m to ~0.00119m
3. **Fine-tune wrist X/Y/rotation** with a calibration target visible to both cameras
4. **Re-record episodes** with corrected TF statics
5. **Verify**: `pcd_z_119mm_fix.html` pipeline — both clouds should fully overlap
6. **Re-extract, re-zarr, re-train** once calibration is confirmed
