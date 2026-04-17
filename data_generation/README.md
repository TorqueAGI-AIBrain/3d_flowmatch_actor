# Data Generation

Tools for collecting real robot demonstrations and converting them into training data for 3D FlowMatch Actor.

## Pipeline

```
                  Option A: Live recording
ROS2 topics ──────► ros2_record_demos.py ──► episode dirs ─┐
                                                            ├──► real_arm_to_zarr.py ──► zarr stores
ROS2 bags (.mcap) ► bag_to_episodes.py ────► episode dirs ─┘
                  Option B: Offline extraction
```

Both paths produce the same episode directory format, which `real_arm_to_zarr.py` converts to zarr for training.

## Step 1: Collect Demonstrations

### Option A: Live Recording

`ros2_record_demos.py` — a ROS2 node that captures synchronized RGB, depth, EEF pose, and gripper state in real time.

```bash
# Configure topics and cameras in the YAML
ros2 run data_generation ros2_record_demos \
    --ros-args \
    -p task_name:=place_wrench \
    -p output_dir:=/data/robot_demos \
    -p cameras:="['front', 'wrist']" \
    -p hz:=10.0 \
    -p instruction:="place the wrench in the toolbox"
```

See `configs/recording.yaml` for the full parameter reference.

**Controls:**
- `ENTER` — start/stop episode recording
- `q` — quit and save

Episodes can also be triggered via ROS2 services:
```bash
ros2 service call /recorder/start_episode std_srvs/srv/Trigger
ros2 service call /recorder/stop_episode  std_srvs/srv/Trigger
```

### Option B: Offline Bag Extraction

`bag_to_episodes.py` — converts pre-recorded ROS2 bag files (`.mcap` or `.db3`) into episode directories. Handles mixed camera setups (e.g. Azure Kinect front + RealSense D435i wrist).

```bash
python -m data_generation.bag_to_episodes --config configs/extraction.yaml
```

Override any config value via CLI:
```bash
python -m data_generation.bag_to_episodes --config configs/extraction.yaml \
    --bag_dir /other/path --target_hz 5.0
```

Expected bag layout:
```
<bag_dir>/
    <task_name>/
        episode_0.mcap
        episode_1.mcap
        ...
        instructions.json  (optional)
```

See `configs/extraction.yaml` for camera topics, sync tolerance, and task list.

## Episode Directory Format

Both collection methods produce this structure:

```
<output_dir>/<task_name>/
    episode_0/
        rgb/front_0000.png          # 256x256 RGB
        rgb/wrist_0000.png
        depth/front_0000.png        # 16-bit PNG, millimeters
        depth/wrist_0000.png
        eef_states.npy              # (T, 8) float32 [x,y,z, qx,qy,qz,qw, gripper]
        camera_extrinsics.npy       # (ncam, 4, 4) cam-to-world
        camera_intrinsics.npy       # (ncam, 3, 3)
    episode_1/
        ...
    instructions.json               # {"0": ["place the wrench in the toolbox"]}
```

## Step 2: Convert to Zarr

Convert episode directories to zarr stores for training:

```bash
python -m data_processing.real_arm_to_zarr \
    --root /data/robot_demos \
    --tgt /data/zarr_output \
    --cameras front wrist \
    --tasks place_wrench \
    --val_ratio 0.1
```

Produces `train.zarr/` and `val.zarr/` compatible with `RealArmDataset`.

## Step 3: Train

```bash
bash scripts/real_arm/train_real_arm.sh
```

Update `DATA_PATH` in the script to point to your zarr output.

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/recording.yaml` | Camera topics, robot topics, recording Hz for live demos |
| `configs/extraction.yaml` | Bag dir, camera topics, sync tolerance, task list for offline extraction |

## Dependencies

Core: `numpy`, `opencv-python`, `scipy`, `pyyaml`

For live recording: `rclpy`, `cv_bridge`, `tf2_ros`, `sensor_msgs`, `geometry_msgs`

For bag extraction: `rosbags==0.9.23`
