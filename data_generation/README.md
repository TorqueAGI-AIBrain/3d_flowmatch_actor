# Data Generation

Tools for collecting real robot demonstrations and converting them into training data for 3D FlowMatch Actor.

## Pipeline

```
ROS2 bags (.mcap) -> bag_to_episodes.py -> episode dirs -> xarm_to_zarr.py -> zarr stores
                                               |
                                        episode_viewer.py (verification)
```

## Bag to Episode Extraction

`bag_to_episodes.py` converts ROS2 bag files (MCAP v9/Jazzy) into episode directories. Uses `mcap.reader` directly (not rosbags AnyReader) for v9 bag support.

```bash
# Extract all episodes
python -m data_generation.bag_to_episodes --config configs/extraction.yaml

# Extract specific episodes (for debugging)
python -m data_generation.bag_to_episodes --config configs/extraction.yaml --episodes 0

# Override config values via CLI
python -m data_generation.bag_to_episodes --config configs/extraction.yaml \
    --bag_dir /other/path --target_hz 5.0 --episodes 0,5,10
```

### Camera Extrinsics

Two modes depending on camera config:

**Static** (fixed camera, e.g. front): looked up directly from `/tf_static` as `world_frame -> cam_frame`. Extrinsics are constant across the episode.

**Dynamic TF chain** (eye-in-hand camera, e.g. wrist): specified as `tf_chain` in config. Joint TFs from `/tf` are time-synced to each frame, chained with static links from `/tf_static`. Produces per-frame extrinsics `(T, ncam, 4, 4)`.

```yaml
# configs/extraction.yaml
cameras:
  front:
    tf_frame: front_optical_frame       # static lookup
  wrist:
    tf_frame: wrist_optical_frame
    tf_chain:                            # dynamic TF chain
      - [link_base, link1]
      - [link1, link2]
      - ...
      - [link_eef, wrist_optical_frame]
```

TF time synchronization is validated per episode — a warning is printed if the closest joint TF is further than `sync_slop` from the target sample time.

### Depth Handling

- Source bags: 32FC1 (float32, metres) or 16UC1 (uint16, mm)
- Resize: `INTER_NEAREST` for depth (no interpolation artifacts), `INTER_AREA` for RGB
- Episode PNGs: uint16 millimetres (`depth_m * 1000`)
- Intrinsics K is scaled to match the 256x256 resize

### Expected Bag Layout

```
<bag_dir>/<task_name>/
    episode_0_bag/
        episode_0_bag_0.mcap
        metadata.yaml
    episode_1_bag/
        ...
```

## Episode Directory Format

```
<output_dir>/<task_name>/
    episode_0/
        rgb/front_0000.png              # 256x256 RGB (uint8)
        rgb/wrist_0000.png
        depth/front_0000.png            # 256x256 depth (uint16 mm)
        depth/wrist_0000.png
        eef_states.npy                  # (T, 8) float32 [x,y,z, qx,qy,qz,qw, gripper]
        camera_extrinsics.npy           # (T, ncam, 4, 4) or (ncam, 4, 4) cam-to-world
        camera_intrinsics.npy           # (ncam, 3, 3) scaled to 256x256
    instructions.json                   # {"0": ["task instruction"]}
```

## Episode Verification

`episode_viewer.py` loads an extracted episode and visualizes it in Open3D to verify that:
- Point clouds from both cameras align in world frame (extrinsics are correct)
- Depth unprojection produces clean geometry (no interpolation artifacts)
- EEF trajectory overlays correctly in the scene
- link_base origin sits at the robot base plate

```bash
# Visualize a single frame
python -m data_generation.episode_viewer \
    --episode_dir data/xarm/.../episode_0 \
    --frame 0

# Multiple frames
python -m data_generation.episode_viewer \
    --episode_dir data/xarm/.../episode_0 \
    --frame 0 --frame 50 --frame 100
```

Displays:
- RGB point clouds from all cameras (transformed to link_base)
- Coordinate axes at link_base origin (R=X, G=Y, B=Z, 15cm)
- EEF trajectory (green line) with current position (yellow sphere)
- Camera positions (red=front, blue=wrist)

## Zarr Conversion

See `data_processing/xarm_to_zarr.py` and `configs/zarr.yaml`.

## Dependencies

- `mcap` (MCAP reader)
- `rosbags` (CDR deserialization via `rosbags.typesys`)
- `numpy`, `opencv-python`, `scipy`
- `open3d` (for episode_viewer only)
