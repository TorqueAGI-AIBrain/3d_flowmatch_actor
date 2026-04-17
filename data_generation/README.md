# Data Generation

Tools for collecting and converting real robot arm demonstrations into training data for 3D FlowMatch Actor.

## Pipeline Overview

```
ROS2 topics ──► ros2_record_demos.py ──► episode directories ──► real_arm_to_zarr.py ──► zarr stores ──► training
```

## 1. Record Demonstrations

`ros2_record_demos.py` is a ROS2 node that captures synchronized RGB, depth, EEF pose, and gripper state from a real robot arm.

### Prerequisites

- ROS2 Humble
- Cameras publishing RGB + depth (e.g. RealSense D435i, Azure Kinect)
- Robot arm publishing EEF pose as `geometry_msgs/PoseStamped`
- TF tree with camera frames for extrinsic calibration

### Usage

```bash
ros2 run data_processing ros2_record_demos \
    --ros-args \
    -p task_name:=pick_cup \
    -p output_dir:=/path/to/demos \
    -p cameras:="['front', 'wrist']" \
    -p hz:=10.0 \
    -p instruction:="pick up the red cup"
```

### Controls

| Input | Action |
|-------|--------|
| ENTER | Start/stop episode recording |
| q | Quit and save |

Episodes can also be triggered via ROS2 services:
```bash
ros2 service call /recorder/start_episode std_srvs/srv/Trigger
ros2 service call /recorder/stop_episode  std_srvs/srv/Trigger
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_name` | `default_task` | Task label for output directory |
| `output_dir` | `/tmp/robot_demos` | Root output path |
| `cameras` | `['front', 'wrist']` | Camera names |
| `hz` | `10.0` | Recording frequency |
| `instruction` | `"do the task"` | Language instruction for the task |
| `eef_topic` | `/end_effector_pose` | EEF pose topic |
| `gripper_topic` | `/gripper/state` | Gripper state topic |
| `world_frame` | `base_link` | TF world frame |

Per-camera topics default to `/camera_<cam>/color/image_raw`, etc. Override with `<cam>_rgb_topic`, `<cam>_depth_topic`, `<cam>_info_topic` parameters.

### Output Structure

```
<output_dir>/<task_name>/
    episode_0/
        rgb/<cam>_0000.png          # 256x256 RGB
        depth/<cam>_0000.png        # 16-bit PNG depth (millimeters)
        eef_states.npy              # (T, 8) [x,y,z, qx,qy,qz,qw, gripper_open]
        camera_extrinsics.npy       # (ncam, 4, 4) cam-to-world
        camera_intrinsics.npy       # (ncam, 3, 3)
    episode_1/
        ...
    instructions.json               # {"0": ["pick up the red cup"]}
```

## 2. Convert to Zarr

Use `data_processing/real_arm_to_zarr.py` to convert episode directories into zarr stores for training:

```bash
python -m data_processing.real_arm_to_zarr \
    --root /path/to/demos \
    --tgt /path/to/zarr_output \
    --cameras front wrist \
    --tasks pick_cup \
    --val_ratio 0.1
```

This produces `train.zarr/` and `val.zarr/` compatible with `RealArmDataset`.

## 3. Train

```bash
bash scripts/real_arm/train_real_arm.sh
```

Update the `DATA_PATH` in the script to point to your zarr output directory.
