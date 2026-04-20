# horizon_eval.py: Evaluate VLA model with GT-anchored autoregressive rollout.
# horizon_eval.py: Hops through GT keyposes, predicts m steps ahead, measures drift.

"""
Evaluation protocol:
  1. Anchor at GT keypose N=0 (GT RGBD + calib + EEF pose)
  2. Autoregressively predict m=5 future waypoints (same RGBD, update proprio)
  3. Compute error between predicted and GT poses at indices 1..m
  4. Hop to GT keypose N=m (fresh GT observation)
  5. Repeat until end of episode

Outputs:
  - 3D matplotlib plots per episode (GT trajectory + predicted segments)
  - Per-segment error metrics (JSON)
  - Data saved as .npz for future visualization (Open3D, Foxglove, etc.)

Usage:
    python -m offline_evaluation.horizon_eval --config configs/evaluation.yaml
"""

import argparse
import json
import os
from collections import deque
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore

from modeling.policy import fetch_model_class
from modeling.encoder.text import fetch_tokenizers
from utils.config import load_yaml_config, flatten_config
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud


IM_SIZE = 256
TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
deserialize_cdr = TYPESTORE.deserialize_cdr


# ---------------------------------------------------------------------------
# MCAP reading
# ---------------------------------------------------------------------------

def _read_mcap_messages(mcap_path, topic_filter=None):
    result = {}
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if topic_filter and channel.topic not in topic_filter:
                continue
            topic = channel.topic
            if topic not in result:
                result[topic] = []
            msg = deserialize_cdr(message.data, schema.name)
            result[topic].append((message.log_time / 1e9, msg))
    return result


def _find_closest_msg(target_time, messages, slop):
    best, best_diff = None, float('inf')
    for t, msg in messages:
        diff = abs(t - target_time)
        if diff < best_diff and diff <= slop:
            best_diff, best = diff, msg
    return best


def _decode_image(msg):
    h, w = msg.height, msg.width
    enc = msg.encoding
    if enc in ('rgb8', 'RGB8'):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
    elif enc in ('bgr8', 'BGR8'):
        return cv2.cvtColor(np.frombuffer(msg.data, np.uint8).reshape(h, w, 3), cv2.COLOR_BGR2RGB)
    elif enc == '16UC1':
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    elif enc == '32FC1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
    elif enc in ('bgra8', 'BGRA8'):
        return cv2.cvtColor(np.frombuffer(msg.data, np.uint8).reshape(h, w, 4), cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported encoding: {enc}")


def _decode_pose(msg):
    p, q = msg.pose.position, msg.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)


def _decode_gripper(msg):
    val = float(msg.data)
    return np.float32(val / 255.0 if val > 1.0 else val)


def _decode_camera_info(msg):
    return np.array(msg.k, dtype=np.float32).reshape(3, 3)


def pose_7d_to_4x4(pose):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def scale_intrinsics(K, orig_w, orig_h, target):
    K = K.copy()
    K[0, 0] *= target / orig_w
    K[1, 1] *= target / orig_h
    K[0, 2] *= target / orig_w
    K[1, 2] *= target / orig_h
    return K


# ---------------------------------------------------------------------------
# Config & model
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Horizon-based offline evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--bag_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="offline_evaluation/results")
    parser.add_argument("--episodes", type=int, nargs="+", default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=5,
                        help="Autoregressive rollout steps per anchor")
    return parser.parse_args()


def load_config(args):
    cfg = load_yaml_config(args.config)
    flat = flatten_config(cfg)
    if args.checkpoint:
        flat['checkpoint'] = args.checkpoint
    if args.bag_dir:
        flat['bag_dir'] = args.bag_dir
    if args.instruction:
        flat['instruction'] = args.instruction
    flat['topics'] = cfg.get('topics', {})
    flat['val_episodes'] = args.episodes or cfg.get('val_episodes', [9, 20, 32])
    flat['output_dir'] = args.output_dir
    flat['horizon'] = args.horizon
    return flat


def load_model(cfg, device):
    model_cls = fetch_model_class(cfg['model_type'])
    model = model_cls(
        backbone=cfg['backbone'],
        finetune_backbone=cfg.get('finetune_backbone', False),
        finetune_text_encoder=cfg.get('finetune_text_encoder', False),
        num_vis_instr_attn_layers=cfg['num_vis_instr_attn_layers'],
        fps_subsampling_factor=cfg['fps_subsampling_factor'],
        embedding_dim=cfg['embedding_dim'],
        num_attn_heads=cfg['num_attn_heads'],
        nhist=cfg['num_history'],
        nhand=2 if cfg.get('bimanual', False) else 1,
        num_shared_attn_layers=cfg['num_shared_attn_layers'],
        relative=cfg.get('relative_action', False),
        rotation_format=cfg['rotation_format'],
        denoise_timesteps=cfg['denoise_timesteps'],
        denoise_model=cfg['denoise_model'],
    )
    ckpt = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    weights = {k.replace('module.', ''): v for k, v in ckpt['weight'].items()}
    model.load_state_dict(weights, strict=False)
    model = model.to(device).eval()
    print(f"Loaded: {cfg['checkpoint']} (iter {ckpt.get('iter', '?')})")
    return model


# ---------------------------------------------------------------------------
# Read episode bag
# ---------------------------------------------------------------------------

def read_episode_bag(bag_dir, topics, target_hz=10.0, sync_slop=0.05):
    bag_dir = Path(bag_dir)
    mcap_files = list(bag_dir.glob('*.mcap'))
    if not mcap_files:
        return None
    msgs = _read_mcap_messages(str(mcap_files[0]), set(topics.values()))

    eef_msgs = msgs.get(topics['eef_pose'], [])
    if len(eef_msgs) < 2:
        return None

    t_start, t_end = eef_msgs[0][0], eef_msgs[-1][0]
    sample_times = np.arange(t_start, t_end, 1.0 / target_hz)
    if len(sample_times) < 2:
        return None

    front_E = pose_7d_to_4x4(_decode_pose(msgs[topics['front_pose']][0][1]))
    front_info = msgs[topics['front_info']][0][1]
    wrist_info = msgs[topics['wrist_info']][0][1]
    first_front_rgb = msgs[topics['front_rgb']][0][1]
    first_wrist_rgb = msgs[topics['wrist_rgb']][0][1]
    front_K = scale_intrinsics(_decode_camera_info(front_info),
                               first_front_rgb.width, first_front_rgb.height, IM_SIZE)
    wrist_K = scale_intrinsics(_decode_camera_info(wrist_info),
                               first_wrist_rgb.width, first_wrist_rgb.height, IM_SIZE)

    gripper_msgs = msgs.get(topics['gripper'], [])
    frames = []
    for t in sample_times:
        eef = _find_closest_msg(t, eef_msgs, sync_slop * 2)
        fr = _find_closest_msg(t, msgs[topics['front_rgb']], sync_slop)
        fd = _find_closest_msg(t, msgs[topics['front_depth']], sync_slop)
        wr = _find_closest_msg(t, msgs[topics['wrist_rgb']], sync_slop)
        wd = _find_closest_msg(t, msgs[topics['wrist_depth']], sync_slop)
        wp = _find_closest_msg(t, msgs[topics['wrist_pose']], sync_slop)
        gr = _find_closest_msg(t, gripper_msgs, sync_slop * 2)
        if any(x is None for x in [eef, fr, fd, wr, wd, wp]):
            continue
        pose = _decode_pose(eef)
        gripper = _decode_gripper(gr) if gr else np.float32(0.0)
        frames.append({
            'front_rgb': _decode_image(fr),
            'front_depth': _decode_image(fd),
            'wrist_rgb': _decode_image(wr),
            'wrist_depth': _decode_image(wd),
            'wrist_E': pose_7d_to_4x4(_decode_pose(wp)),
            'eef_pose': np.concatenate([pose, [gripper]]),
            'timestamp': t - t_start,
        })
    if len(frames) < 2:
        return None
    return {'frames': frames, 'front_E': front_E, 'front_K': front_K, 'wrist_K': wrist_K}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame, front_E, front_K, wrist_K, eef_history,
                     depth2cloud, depth_scale, device):
    f_rgb = cv2.resize(frame['front_rgb'], (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_AREA)
    w_rgb = cv2.resize(frame['wrist_rgb'], (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_AREA)
    rgbs = np.stack([
        f_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0,
        w_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0,
    ])
    rgbs = torch.from_numpy(rgbs).unsqueeze(0).to(device)

    f_d = cv2.resize(frame['front_depth'].astype(np.float32), (IM_SIZE, IM_SIZE),
                     interpolation=cv2.INTER_NEAREST) / depth_scale
    w_d = cv2.resize(frame['wrist_depth'].astype(np.float32), (IM_SIZE, IM_SIZE),
                     interpolation=cv2.INTER_NEAREST) / depth_scale
    depth = torch.from_numpy(np.stack([f_d, w_d])).unsqueeze(0).to(device)

    extr = torch.from_numpy(np.stack([front_E, frame['wrist_E']])).unsqueeze(0).to(device)
    intr = torch.from_numpy(np.stack([front_K, wrist_K])).unsqueeze(0).to(device)
    pcds = depth2cloud(depth, extr, intr)

    proprio = torch.from_numpy(
        np.stack(list(eef_history))
    ).unsqueeze(0).unsqueeze(2).to(device)

    return rgbs, pcds, proprio


def predict_one_step(model, rgbs, pcds, proprio, instr_tokens, device):
    action_mask = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = model(None, action_mask, rgbs, None, pcds,
                     instr_tokens, proprio[:, :, :, :7], run_inference=True)
    return pred[0, 0, 0].cpu().float().numpy()


# ---------------------------------------------------------------------------
# Horizon evaluation
# ---------------------------------------------------------------------------

def evaluate_episode(ep_idx, cfg, model, tokenizer, depth2cloud, device):
    bag_dir = Path(cfg['bag_dir']) / f"episode_{ep_idx}_bag"
    if not bag_dir.exists():
        print(f"  Bag not found: {bag_dir}")
        return None

    episode = read_episode_bag(bag_dir, cfg['topics'],
                               cfg.get('target_hz', 10.0), cfg.get('sync_slop', 0.05))
    if episode is None:
        return None

    frames = episode['frames']
    N = len(frames)
    m = cfg['horizon']
    num_history = cfg['num_history']
    depth_scale = cfg.get('depth_scale', 1000.0)
    instr_tokens = tokenizer([cfg.get('instruction', 'do the task')]).to(device)

    print(f"  {N} frames, horizon m={m}, anchors every {m} steps")

    gt_all = np.array([f['eef_pose'][:7] for f in frames])
    segments = []
    anchor_indices = list(range(0, N - 1, m))

    for anchor in tqdm(anchor_indices, desc="  Segments"):
        eef_history = deque(maxlen=num_history)
        anchor_state = frames[anchor]['eef_pose']
        for _ in range(num_history):
            eef_history.append(anchor_state.copy())

        rgbs, pcds, _ = preprocess_frame(
            frames[anchor], episode['front_E'], episode['front_K'],
            episode['wrist_K'], eef_history, depth2cloud, depth_scale, device,
        )

        seg_preds, seg_gt, seg_errors = [], [], []
        for step in range(m):
            target_idx = anchor + step + 1
            if target_idx >= N:
                break

            proprio = torch.from_numpy(
                np.stack(list(eef_history))
            ).unsqueeze(0).unsqueeze(2).to(device)

            pred_np = predict_one_step(model, rgbs, pcds, proprio, instr_tokens, device)
            gt_pose = gt_all[target_idx]

            eef_history.append(pred_np.copy())

            pos_err = np.linalg.norm(pred_np[:3] - gt_pose[:3])
            seg_preds.append(pred_np[:7].copy())
            seg_gt.append(gt_pose.copy())
            seg_errors.append(pos_err)

        if seg_preds:
            segments.append({
                'anchor_idx': anchor,
                'anchor_pos': gt_all[anchor, :3].copy(),
                'preds': np.array(seg_preds),
                'gt': np.array(seg_gt),
                'errors': np.array(seg_errors),
            })

    return {
        'episode': ep_idx, 'num_frames': N, 'horizon': m,
        'gt_all': gt_all, 'segments': segments,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_horizon_eval(result, output_dir):
    ep_idx = result['episode']
    gt_all = result['gt_all']
    segments = result['segments']
    m = result['horizon']

    fig = plt.figure(figsize=(18, 7))
    gt_pos = gt_all[:, :3]
    N = len(gt_pos)
    cmap_gt = plt.cm.Greens(np.linspace(0.3, 1.0, N))

    # 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
                c=cmap_gt, s=8, alpha=0.6, depthshade=False, label='GT')

    colors = plt.cm.Reds(np.linspace(0.4, 1.0, len(segments)))
    for i, seg in enumerate(segments):
        ax1.scatter(*seg['anchor_pos'], color='#00ff88', s=60, marker='^',
                    edgecolors='k', linewidths=0.5, zorder=10)
        preds = seg['preds'][:, :3]
        full = np.vstack([seg['anchor_pos'][None], preds])
        ax1.plot(full[:, 0], full[:, 1], full[:, 2], '-o',
                 color=colors[i], markersize=4, linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Episode {ep_idx} — 3D (m={m})')
    ax1.view_init(elev=25, azim=-60)

    # XY projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(gt_pos[:, 0], gt_pos[:, 1], c=cmap_gt, s=8, alpha=0.6, label='GT')
    for i, seg in enumerate(segments):
        preds = seg['preds'][:, :3]
        full = np.vstack([seg['anchor_pos'][None], preds])
        ax2.plot(full[:, 0], full[:, 1], '-o', color=colors[i],
                 markersize=4, linewidth=1.5, alpha=0.8)
        ax2.plot(seg['anchor_pos'][0], seg['anchor_pos'][1], '^',
                 color='#00ff88', markersize=6, zorder=10)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Error by step
    ax3 = fig.add_subplot(133)
    step_errors = {i: [] for i in range(m)}
    for seg in segments:
        for j, err in enumerate(seg['errors']):
            step_errors[j].append(err * 100)
    steps = sorted(step_errors.keys())
    means = [np.mean(step_errors[s]) if step_errors[s] else 0 for s in steps]
    stds = [np.std(step_errors[s]) if step_errors[s] else 0 for s in steps]
    ax3.bar([s + 1 for s in steps], means, yerr=stds, capsize=3,
            color='#3498db', alpha=0.7, edgecolor='#2c3e50')
    ax3.set_xlabel('Step within horizon')
    ax3.set_ylabel('Position Error (cm)')
    ax3.set_title('Error vs Rollout Step')
    ax3.set_xticks([s + 1 for s in steps])
    ax3.grid(True, alpha=0.3, axis='y')

    all_errors = np.concatenate([s['errors'] for s in segments]) * 100
    plt.suptitle(
        f'Episode {ep_idx} | {result["num_frames"]} frames | '
        f'horizon={m} | {len(segments)} segments\n'
        f'Mean err: {all_errors.mean():.2f}cm | Max err: {all_errors.max():.2f}cm | '
        f'Step-1 err: {means[0]:.2f}cm | Step-{m} err: {means[-1]:.2f}cm',
        fontsize=12, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'episode_{ep_idx}_horizon_m{m}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading model...")
    model = load_model(cfg, device)
    tokenizer = fetch_tokenizers(cfg['backbone'])
    depth2cloud = RLBenchDepth2Cloud((IM_SIZE, IM_SIZE))

    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for ep_idx in cfg['val_episodes']:
        print(f"\n=== Episode {ep_idx} ===")
        result = evaluate_episode(ep_idx, cfg, model, tokenizer, depth2cloud, device)
        if result is None:
            continue
        plot_horizon_eval(result, output_dir)

        npz_path = os.path.join(output_dir, f'episode_{ep_idx}_horizon_m{cfg["horizon"]}.npz')
        np.savez_compressed(
            npz_path, gt_all=result['gt_all'],
            **{f'seg{i}_anchor': s['anchor_pos'] for i, s in enumerate(result['segments'])},
            **{f'seg{i}_preds': s['preds'] for i, s in enumerate(result['segments'])},
            **{f'seg{i}_gt': s['gt'] for i, s in enumerate(result['segments'])},
            **{f'seg{i}_errors': s['errors'] for i, s in enumerate(result['segments'])},
        )
        print(f"  Data saved: {npz_path}")

        seg_summaries = []
        for seg in result['segments']:
            seg_summaries.append({
                'anchor_idx': seg['anchor_idx'],
                'num_steps': len(seg['errors']),
                'mean_error_cm': float(seg['errors'].mean() * 100),
                'max_error_cm': float(seg['errors'].max() * 100),
                'per_step_error_cm': [float(e * 100) for e in seg['errors']],
            })
        all_results.append({
            'episode': ep_idx, 'horizon': cfg['horizon'],
            'num_frames': result['num_frames'],
            'num_segments': len(result['segments']),
            'segments': seg_summaries,
        })

    metrics_path = os.path.join(output_dir, f'metrics_m{cfg["horizon"]}.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")
    print("Done.")


if __name__ == '__main__':
    main()
