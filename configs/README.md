# Configs

YAML configuration files for each stage of the data-to-evaluation pipeline.

## Pipeline Stages

### 1. Extract episodes from MCAP bags

```bash
docker run --rm -v $(pwd):/app -w /app 3dfa \
  python3 -m data_generation.bag_to_episodes --config configs/extraction.yaml
```

Config: [`extraction.yaml`](extraction.yaml) — bag directory, output directory, camera topics, robot topics, image size, depth scale, sync tolerance.

### 2. Convert episodes to zarr

```bash
docker run --rm -v $(pwd):/app -w /app 3dfa \
  python3 -m data_processing.xarm_to_zarr --config configs/zarr.yaml
```

Config: [`zarr.yaml`](zarr.yaml) — episode root, output path, camera names, quaternion format, val split ratio, trajectory length, keyframe detection.

### 3. Train

```bash
docker run --rm --gpus all -v $(pwd):/app -w /app 3dfa \
  torchrun --nproc_per_node 1 --master_port $RANDOM \
  main.py --config configs/training.yaml
```

Config: [`training.yaml`](training.yaml) — dataset paths, model architecture, training hyperparameters.

### 4. Evaluate (offline)

```bash
docker run --rm --gpus all -v $(pwd):/app -w /app 3dfa \
  python3 -m offline_evaluation.horizon_eval --config configs/evaluation.yaml --horizon 2
```

Config: [`evaluation.yaml`](evaluation.yaml) — checkpoint path, bag directory, val episodes, topic mapping, model params.

### 5. Visualize results

```bash
docker run --rm -p 7860:7860 -v $(pwd):/app -w /app 3dfa \
  python3 -m offline_evaluation.visualize --results_dir offline_evaluation/results
```

No config needed — reads `.npz` files from the results directory.

### 6. Inference (real robot, ROS2)

```bash
python3 inference/inference_node.py --config configs/inference.yaml
```

Config: [`inference.yaml`](inference.yaml) — checkpoint, camera/robot ROS2 topics, workspace bounds.
