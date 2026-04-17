# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D FlowMatch Actor (3DFA) is a 3D visual policy for robot manipulation that combines flow matching with 3D scene representations. It supports single-arm (xArm) and dual-arm (PerAct2 bimanual) manipulation. This repo also includes a re-implementation of 3D Diffuser Actor (3DDA).

## Common Commands

### Training (DDP via torchrun)
```bash
# xArm real robot (uses YAML config)
bash scripts/xarm/train.sh configs/training.yaml

# RLBench simulation benchmarks
bash scripts/rlbench/train_peract2.sh
bash scripts/rlbench/train_peract.sh
bash scripts/rlbench/train_hiveformer.sh

# Direct torchrun invocation (YAML + CLI overrides)
torchrun --nproc_per_node 1 --master_port $RANDOM main.py --config configs/training.yaml --batch_size 8
```

### Inference (ROS2 node for xArm deployment)
```bash
python inference/inference_node.py --config configs/inference.yaml
```

### Online Evaluation (RLBench simulator)
```bash
bash online_evaluation_rlbench/eval_peract2.sh
bash online_evaluation_rlbench/eval_peract.sh
bash online_evaluation_rlbench/eval_hiveformer.sh
```

### Data Pipeline (xArm real robot)
```
ROS2 bag → bag_to_episodes.py → episode directories → xarm_to_zarr.py → zarr store
```
Configs: `configs/extraction.yaml`, `configs/zarr.yaml`

### Docker
```bash
docker build -t 3dfa .
# Base image: ghcr.io/torqueagi-aibrain/transformers-pytorch-gpu:2025-10-02
```

### Installation (conda, no Docker)
```bash
conda create -y --name 3dfa python=3.10
conda activate 3dfa
pip install torch torchvision torchaudio
pip install einops tqdm transformers zarr diffusers kornia tensorboard
pip install -e .
```

## Architecture

### Configuration System
YAML-first config with CLI overrides. `main.py` defines `ARGUMENT_SPEC` as the canonical list of all arguments. YAML values are loaded as defaults, CLI args take precedence. Config loading: `utils/config.py`.

### Model Pipeline
1. **Vision-Language Encoder** (`modeling/encoder/`) — CLIP backbone + FPN extracts visual features, CLIP text encoder processes language instructions. Point clouds are constructed from depth via `utils/depth2cloud/`.
2. **Policy Head** (`modeling/policy/`) — Transformer decoder with 3D relative attention between action tokens and visual tokens. Two variants: `DenoiseActor3D` (standard) and `DenoiseActor2D`.
3. **Noise Scheduler** (`modeling/noise_scheduler/`) — Three options: `rectified_flow` (default, 5-10 steps), `ddpm`, `ddim`. Rectified flow is 10-30x faster than diffusion.

Model selection: `--model_type denoise3d|denoise2d`, `--denoise_model rectified_flow|ddpm|ddim`

### Datasets
Registry in `datasets/__init__.py` via `fetch_dataset_class()`. All datasets use zarr for fast loading.

| `--dataset` value | Class | Use case |
|---|---|---|
| `XArm` | `XArmDataset` | Real xArm 7DOF robot |
| `Peract2_3dfront_3dwrist` | `Peract2Dataset` | Bimanual (2 cameras) |
| `Peract2_3dfront` | `Peract2SingleCamDataset` | Bimanual (1 camera) |
| `Peract` | `PeractDataset` | Unimanual RLBench |
| `HiveformerRLBench` | `HiveformerDataset` | 74-task trajectory |

### Training Loop
`utils/trainers/base.py` contains the core train/eval loop. Subclasses in `utils/trainers/rlbench.py` and `utils/trainers/peract.py` handle benchmark-specific data preprocessing. Training always uses DDP (`torch.distributed`), even single-GPU. The trainer encodes the scene once, then denoises multiple times per step (`lv2_batch_size`).

Logs go to `train_logs/<exp_log_dir>/<run_log_dir>/` with TensorBoard events and periodic checkpoints.

### Inference (Real Robot)
`inference/inference_node.py` is a ROS2 node subscribing to camera RGB/depth topics and robot state, running the model, and publishing predicted EEF poses. Configured entirely via `configs/inference.yaml` (camera topics, robot topics, workspace bounds).

## Key Design Patterns

- **Action representation**: positions (3D) + rotation (quaternion xyzw converted internally to 6D) + gripper (1D). Shape: `(bs, num_trajectory_steps, num_arms, 8)`.
- **Workspace normalization**: actions are normalized to [-1, 1] using workspace bounds + a buffer (`workspace_normalizer_buffer`).
- **Rotation format**: `quat_xyzw` default, converted to 6D rotation internally. Euler also supported via `--rotation_format`.
- **No tests**: this repo has no test suite. Validation happens through training val loss and online RLBench evaluation.

## Important Defaults

- `denoise_timesteps: 10` (5 for xArm config)
- `embedding_dim: 120` (must be divisible by `num_attn_heads`)
- `backbone: clip`, `finetune_backbone: false`
- `keypose_only: true` (predict keyposes, not full trajectories; set false for HiveFormer)
- `batch_size: 64` for RLBench, `16` for xArm
- `train_iters: 600000` default, `100000` for xArm
