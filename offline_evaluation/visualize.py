# visualize.py: Interactive 3D visualization of horizon evaluation results.
# visualize.py: Gradio app with Plotly 3D scatter, PCD + trajectory overlay.

"""
Launch an interactive Gradio UI to explore horizon evaluation results.
Loads .npz files from offline_evaluation/results/ and renders 3D plots
with Plotly (rotatable, zoomable). Main view overlays predicted trajectory
on the scene point cloud.

Usage:
    python -m offline_evaluation.visualize --results_dir offline_evaluation/results
"""

import argparse
import glob
import os
import re

import gradio as gr
import numpy as np
import plotly.graph_objects as go


def load_results(results_dir):
    """Load all .npz result files, return dict keyed by 'ep{idx}_m{horizon}'."""
    results = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
        fname = os.path.basename(path)
        match = re.match(r'episode_(\d+)_horizon_m(\d+)\.npz', fname)
        if not match:
            continue
        ep_idx = int(match.group(1))
        horizon = int(match.group(2))
        data = np.load(path, allow_pickle=True)

        gt_all = data['gt_all']
        segments = []
        i = 0
        while f'seg{i}_anchor' in data:
            segments.append({
                'anchor': data[f'seg{i}_anchor'],
                'preds': data[f'seg{i}_preds'],
                'gt': data[f'seg{i}_gt'],
                'errors': data[f'seg{i}_errors'],
            })
            i += 1
        pcd_data = {}
        if 'front_pts' in data:
            pcd_data['front_pts'] = data['front_pts']
            pcd_data['front_rgb'] = data['front_rgb']
            pcd_data['wrist_pts'] = data['wrist_pts']
            pcd_data['wrist_rgb'] = data['wrist_rgb']

        key = f"ep{ep_idx}_m{horizon}"
        results[key] = {
            'gt_all': gt_all, 'segments': segments,
            'horizon': horizon, 'episode': ep_idx,
            'path': path, **pcd_data,
        }
    return results


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_pcd_trajectory_figure(result, show_front=True, show_wrist=True,
                                show_gt=True, show_preds=True,
                                show_error_lines=False):
    """Main view: point cloud with GT + predicted trajectory overlay."""
    fig = go.Figure()
    segments = result['segments']
    horizon = result['horizon']

    # Point clouds
    def _add_pcd(pts, rgb, name, visible):
        if pts is None or not visible:
            return
        colors = [f'rgb({r},{g},{b})' for r, g, b in rgb]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=1.2, color=colors, opacity=0.6),
            name=name,
            hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}',
        ))

    _add_pcd(result.get('front_pts'), result.get('front_rgb'),
             'Front PCD', show_front)
    _add_pcd(result.get('wrist_pts'), result.get('wrist_rgb'),
             'Wrist PCD', show_wrist)

    # GT trajectory
    if show_gt:
        gt_pos = result['gt_all'][:, :3]
        N = len(gt_pos)
        fig.add_trace(go.Scatter3d(
            x=gt_pos[:, 0], y=gt_pos[:, 1], z=gt_pos[:, 2],
            mode='lines+markers',
            line=dict(color='#2ecc71', width=4),
            marker=dict(size=3, color='#2ecc71', opacity=0.6),
            name='GT trajectory',
            hovertemplate='GT %{customdata}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}',
            customdata=list(range(N)),
        ))
        # Anchor markers
        anchors = np.array([s['anchor'] for s in segments])
        fig.add_trace(go.Scatter3d(
            x=anchors[:, 0], y=anchors[:, 1], z=anchors[:, 2],
            mode='markers',
            marker=dict(size=8, color='#00ff88', symbol='diamond',
                        line=dict(color='black', width=1.5)),
            name='GT anchors',
        ))

    # Predicted trajectory overlay
    if show_preds and segments:
        n_segs = len(segments)
        for i, seg in enumerate(segments):
            preds = seg['preds'][:, :3]
            anchor = seg['anchor']
            errors = seg['errors'] * 100
            full_path = np.vstack([anchor[None], preds])

            # Color by error magnitude: green (<0.5cm) → yellow → red (>2cm)
            err_norm = np.clip(errors / 2.0, 0, 1)  # 0-2cm mapped to 0-1
            r_ch = (err_norm * 255).astype(int)
            g_ch = ((1 - err_norm) * 255).astype(int)
            pt_colors = [f'rgb({r},{g},40)' for r, g in zip(r_ch, g_ch)]

            fig.add_trace(go.Scatter3d(
                x=preds[:, 0], y=preds[:, 1], z=preds[:, 2],
                mode='markers',
                marker=dict(size=6, color=pt_colors, opacity=1.0,
                            line=dict(color='black', width=0.5)),
                name=f'Pred {i}' if i < 3 else None,
                showlegend=(i < 3),
                hovertemplate=(
                    f'Seg {i}, step %{{customdata[0]}}<br>'
                    f'err=%{{customdata[1]:.2f}}cm<br>'
                    f'x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}'
                ),
                customdata=list(zip(range(1, len(preds) + 1), errors.tolist())),
            ))

            fig.add_trace(go.Scatter3d(
                x=full_path[:, 0], y=full_path[:, 1], z=full_path[:, 2],
                mode='lines',
                line=dict(color='#e74c3c', width=3),
                showlegend=False, hoverinfo='skip',
            ))

            # Error lines connecting pred to GT
            if show_error_lines:
                gt_seg = seg['gt'][:, :3]
                for j in range(len(preds)):
                    fig.add_trace(go.Scatter3d(
                        x=[preds[j, 0], gt_seg[j, 0]],
                        y=[preds[j, 1], gt_seg[j, 1]],
                        z=[preds[j, 2], gt_seg[j, 2]],
                        mode='lines',
                        line=dict(color='rgba(255,255,0,0.5)', width=1.5,
                                  dash='dot'),
                        showlegend=False, hoverinfo='skip',
                    ))

    ep = result.get('episode', '?')
    fig.update_layout(
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                   aspectmode='data'),
        title=f'Episode {ep} | PCD + Trajectory (m={horizon})',
        width=1000, height=750,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_trajectory_only_figure(result, show_gt_line=True,
                                 show_segments=True, show_error_lines=True):
    """Trajectory-only view without point cloud (lighter, faster rotation)."""
    gt_all = result['gt_all']
    segments = result['segments']
    horizon = result['horizon']
    gt_pos = gt_all[:, :3]
    N = len(gt_pos)

    fig = go.Figure()

    t_norm = np.linspace(0, 1, N)
    fig.add_trace(go.Scatter3d(
        x=gt_pos[:, 0], y=gt_pos[:, 1], z=gt_pos[:, 2],
        mode='markers',
        marker=dict(size=2, color=t_norm, colorscale='Greens',
                    opacity=0.4, showscale=False),
        name='GT (all frames)',
        hovertemplate='GT frame %{customdata}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}',
        customdata=list(range(N)),
    ))

    if show_gt_line:
        fig.add_trace(go.Scatter3d(
            x=gt_pos[:, 0], y=gt_pos[:, 1], z=gt_pos[:, 2],
            mode='lines',
            line=dict(color='rgba(46,204,113,0.3)', width=2),
            name='GT path', showlegend=False,
        ))

    anchors = np.array([s['anchor'] for s in segments])
    fig.add_trace(go.Scatter3d(
        x=anchors[:, 0], y=anchors[:, 1], z=anchors[:, 2],
        mode='markers',
        marker=dict(size=7, color='#00ff88', symbol='diamond',
                    line=dict(color='black', width=1.5)),
        name=f'GT keyposes (every {horizon})',
    ))

    if show_segments:
        n_segs = len(segments)
        seg_colors = [f'hsl({int(i * 360 / n_segs)}, 80%, 55%)'
                      for i in range(n_segs)]
        for i, seg in enumerate(segments):
            preds = seg['preds'][:, :3]
            anchor = seg['anchor']
            errors = seg['errors'] * 100
            full_path = np.vstack([anchor[None], preds])

            fig.add_trace(go.Scatter3d(
                x=preds[:, 0], y=preds[:, 1], z=preds[:, 2],
                mode='markers',
                marker=dict(size=4, color=seg_colors[i], opacity=0.9),
                name=f'Seg {i} pred' if i < 5 else None,
                showlegend=(i < 5),
                hovertemplate=(
                    f'Seg {i}, step %{{customdata[0]}}<br>'
                    f'err=%{{customdata[1]:.2f}}cm<br>'
                    f'x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}'
                ),
                customdata=list(zip(range(1, len(preds) + 1), errors.tolist())),
            ))
            fig.add_trace(go.Scatter3d(
                x=full_path[:, 0], y=full_path[:, 1], z=full_path[:, 2],
                mode='lines', line=dict(color=seg_colors[i], width=3),
                showlegend=False, hoverinfo='skip',
            ))
            if show_error_lines:
                gt_seg = seg['gt'][:, :3]
                for j in range(len(preds)):
                    fig.add_trace(go.Scatter3d(
                        x=[preds[j, 0], gt_seg[j, 0]],
                        y=[preds[j, 1], gt_seg[j, 1]],
                        z=[preds[j, 2], gt_seg[j, 2]],
                        mode='lines',
                        line=dict(color='rgba(200,200,200,0.3)', width=1,
                                  dash='dot'),
                        showlegend=False, hoverinfo='skip',
                    ))

    fig.update_layout(
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                   aspectmode='data'),
        title=f'Trajectory Only (m={horizon}, {len(segments)} segments)',
        width=1000, height=750,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_error_bar_figure(result):
    segments = result['segments']
    horizon = result['horizon']
    step_errors = {i: [] for i in range(horizon)}
    for seg in segments:
        for j, err in enumerate(seg['errors']):
            step_errors[j].append(err * 100)

    steps = list(range(horizon))
    means = [np.mean(step_errors[s]) for s in steps]
    stds = [np.std(step_errors[s]) for s in steps]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'Step {s+1}' for s in steps], y=means,
        error_y=dict(type='data', array=stds, visible=True),
        marker_color='#3498db',
        text=[f'{m:.2f}cm' for m in means], textposition='outside',
    ))
    fig.update_layout(title='Position Error by Rollout Step',
                      yaxis_title='Error (cm)', xaxis_title='Step',
                      height=350)
    return fig


def build_per_segment_figure(result):
    segments = result['segments']
    means = [seg['errors'].mean() * 100 for seg in segments]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(means))), y=means,
        mode='lines+markers',
        marker=dict(size=5, color='#e74c3c'),
        line=dict(color='#e74c3c', width=2),
    ))
    fig.add_hline(y=np.mean(means), line_dash='dash', line_color='gray',
                  annotation_text=f'Mean: {np.mean(means):.2f}cm')
    fig.update_layout(title='Mean Error per Segment',
                      xaxis_title='Segment index', yaxis_title='Mean error (cm)',
                      height=350)
    return fig


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def create_app(results_dir):
    results = load_results(results_dir)
    if not results:
        raise RuntimeError(f"No .npz files found in {results_dir}")

    episode_choices = sorted(results.keys())

    def update_all(key, show_front, show_wrist, show_gt, show_preds,
                   show_err_lines, show_traj_gt, show_traj_segs,
                   show_traj_err):
        r = results[key]
        ep_idx = r.get('episode', key)

        fig_pcd = build_pcd_trajectory_figure(
            r, show_front, show_wrist, show_gt, show_preds, show_err_lines)
        fig_traj = build_trajectory_only_figure(
            r, show_traj_gt, show_traj_segs, show_traj_err)
        fig_bar = build_error_bar_figure(r)
        fig_seg = build_per_segment_figure(r)

        all_errors = np.concatenate([s['errors'] for s in r['segments']]) * 100
        step1_errs = [s['errors'][0] * 100 for s in r['segments']]
        summary = (
            f"**Episode {ep_idx}** | "
            f"{len(r['gt_all'])} frames | horizon={r['horizon']} | "
            f"{len(r['segments'])} segments\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Mean error | {all_errors.mean():.2f} cm |\n"
            f"| Max error | {all_errors.max():.2f} cm |\n"
            f"| Step-1 error | {np.mean(step1_errs):.2f} cm |\n"
            f"| Std | {all_errors.std():.2f} cm |\n"
        )
        return fig_pcd, fig_traj, fig_bar, fig_seg, summary

    with gr.Blocks(title="3DFA Horizon Evaluation",
                   theme=gr.themes.Soft()) as app:
        gr.Markdown("# 3DFA Horizon Evaluation Viewer")
        gr.Markdown(
            "**Left**: Scene point cloud + predicted trajectory overlay. "
            "**Right**: Trajectory-only view (lighter). "
            "Predictions colored green→red by error magnitude."
        )

        with gr.Row():
            ep_dropdown = gr.Dropdown(
                choices=[str(e) for e in episode_choices],
                value=str(episode_choices[0]), label="Episode / Horizon",
            )

        summary_md = gr.Markdown()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### PCD + Trajectory")
                with gr.Row():
                    show_front = gr.Checkbox(value=True, label="Front PCD")
                    show_wrist = gr.Checkbox(value=True, label="Wrist PCD")
                    show_gt = gr.Checkbox(value=True, label="GT traj")
                    show_preds = gr.Checkbox(value=True, label="Predictions")
                    show_err = gr.Checkbox(value=False, label="Error lines")
                plot_pcd = gr.Plot(label="PCD + Trajectory")

            with gr.Column():
                gr.Markdown("### Trajectory Only")
                with gr.Row():
                    show_traj_gt = gr.Checkbox(value=True, label="GT path")
                    show_traj_segs = gr.Checkbox(value=True, label="Segments")
                    show_traj_err = gr.Checkbox(value=False, label="Error lines")
                plot_traj = gr.Plot(label="Trajectory Only")

        with gr.Row():
            plot_bar = gr.Plot(label="Error vs Rollout Step")
            plot_seg = gr.Plot(label="Error per Segment")

        inputs = [ep_dropdown, show_front, show_wrist, show_gt, show_preds,
                  show_err, show_traj_gt, show_traj_segs, show_traj_err]
        outputs = [plot_pcd, plot_traj, plot_bar, plot_seg, summary_md]

        for inp in inputs:
            inp.change(update_all, inputs, outputs)
        app.load(update_all, inputs, outputs)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="offline_evaluation/results")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = create_app(args.results_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
