# TF Correction: Front Camera and Wrist Calibration

## Problem

Point clouds from the front (Azure Kinect) and wrist (RealSense D435i) cameras do not align in `link_base` frame. The TF static transforms published in the bags are incorrect:

1. `link_base -> front_optical_frame`: the `/front/pose` topic's coordinate frame origin is offset from the true `link_base` (robot base plate)
2. `link_eef -> wrist_optical_frame`: Z translation was 11.9cm, should be ~0.19cm; Y and yaw also off

## Corrections Applied

### 1. Front camera: `link_base -> front_optical_frame`

A correction matrix was obtained by exporting the front camera point cloud + coordinate axes to PLY, then aligning the axes to the robot base plate in CloudCompare. Additional fine-tuning offsets were applied from Foxglove verification.

**CloudCompare correction matrix** (applied as `inv(correction) @ /front/pose`):
```
[[0.990103,  0.139081, -0.018772, 0.036359],
 [-0.133884, 0.976160,  0.170840, 0.000056],
 [0.042085, -0.166636,  0.985120, 0.067018],
 [0.000000,  0.000000,  0.000000, 1.000000]]
```

**Additional offsets** (applied after the correction matrix):
- Z: -2cm (moves link_base up to match table surface)
- Y: +2cm

**Both `/tf_static` and `/front/pose`** are set to the same corrected value so TF and pose topics are consistent.

### 2. Wrist camera: `link_eef -> wrist_optical_frame`

Offsets identified from Foxglove visual alignment:

| Parameter | Original | Correction | Final |
|-----------|----------|------------|-------|
| Translation X | 0.063 | unchanged | 0.063 |
| Translation Y | -0.033 | -0.01 | -0.043 |
| Translation Z | 0.119 | set to 0.0119, then -0.01 | 0.002 |
| Yaw (Z rotation) | 87.8 deg | -3 deg | 84.8 deg |

The original Z=11.9cm was ~10x too large. The physical camera mount offset from the EEF flange is ~1-2mm in Z.

## How to Reproduce

### Step 1: Diagnostic PLYs (tf_fixes.py)

Extract front+wrist point clouds from a single MCAP frame, using both `/pose` topics and TF chain, for comparison in a 3D viewer:

```bash
python -m data_processing.tf_fixes
```

Outputs PLY files in `data_processing/debug_pcds/` with both transform variants.

### Step 2: Interactive calibration (calibrate_front.py)

Apply corrections and export PLYs with coordinate axes at link_base origin. Use `--frame` to test at different arm configurations:

```bash
python -m data_processing.calibrate_front --frame 0
python -m data_processing.calibrate_front --frame 100
python -m data_processing.calibrate_front --frame 150
```

Overlay `origin_axes.ply` + `front_calibrated.ply` + `wrist_calibrated.ply` in CloudCompare. Adjust `FRONT_CORRECTION` / `WRIST_CORRECTION` matrices in the script and re-run.

### Step 3: Apply to all bags (apply_tf_corrections.py)

Apply the finalized corrections to all MCAP bags:

```bash
python -m data_processing.apply_tf_corrections \
    --src data/xarm/default_task_fixed \
    --dst data/xarm/default_task_fixed_v3
```

This rewrites:
- `/tf_static`: `link_base -> front_optical_frame` (corrected) and `link_eef -> wrist_optical_frame` (Z/Y/yaw fixed)
- `/front/pose`: set to match corrected TF static (derived from TF, not independent)
- All other topics copied unchanged

### Step 4: Batch verify (verify_tf_batch.py)

Check table-surface Z agreement between front and wrist across all episodes:

```bash
python -m data_processing.verify_tf_batch --bag_dir data/xarm/default_task_fixed_v3
```

### Step 5: Verify in Foxglove

Open corrected bags in Foxglove Studio with `link_base` as fixed frame. The front camera depth point cloud should align with the TF frame axes at the robot base plate. The wrist camera point cloud should overlap with the front camera's view of the same scene.

## Files

| File | Purpose |
|------|---------|
| `data_processing/tf_fixes.py` | Diagnostic: extract PCDs using both /pose and TF, export PLY |
| `data_processing/calibrate_front.py` | Interactive: tune corrections, export PLY with axes |
| `data_processing/apply_tf_corrections.py` | Batch: apply corrections to all MCAP bags |
| `data_processing/verify_tf_batch.py` | Batch: verify table-Z alignment across episodes |
| `docs/tf_calibration_debug.md` | Investigation notes from the debugging process |
