# visualize.py: Interactive 3D visualization of horizon evaluation results.
# visualize.py: Gradio app with Plotly 3D scatter, keyposes highlighted.

"""
Launch an interactive Gradio UI to explore horizon evaluation results.
Loads .npz files from offline_evaluation/results/ and renders 3D plots
with Plotly (rotatable, zoomable). GT keyposes (anchor points) are
highlighted with larger, brighter markers.

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
    """Load all .npz result files, return dict keyed by episode index."""
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
        results[ep_idx] = {
            'gt_all': gt_all, 'segments': segments,
            'horizon': horizon, 'path': path,
        }
    return results


def build_3d_figure(result, show_gt_line=True, show_segments=True,
                    show_error_lines=True):
    """Build a Plotly 3D figure for one episode's horizon evaluation."""
    gt_all = result['gt_all']
    segments = result['segments']
    horizon = result['horizon']
    gt_pos = gt_all[:, :3]
    N = len(gt_pos)

    fig = go.Figure()

    # GT trajectory as small muted dots
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

    # GT keyposes — bright green diamonds
    anchor_positions = np.array([s['anchor'] for s in segments])
    anchor_indices = [i * horizon for i in range(len(segments))]
    fig.add_trace(go.Scatter3d(
        x=anchor_positions[:, 0], y=anchor_positions[:, 1], z=anchor_positions[:, 2],
        mode='markers',
        marker=dict(size=7, color='#00ff88', symbol='diamond',
                    line=dict(color='black', width=1.5)),
        name=f'GT keyposes (every {horizon})',
        hovertemplate='Anchor %{customdata}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}',
        customdata=anchor_indices,
    ))

    # Predicted segments
    if show_segments:
        n_segs = len(segments)
        seg_colors = [f'hsl({int(i * 360 / n_segs)}, 80%, 55%)' for i in range(n_segs)]

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
                mode='lines',
                line=dict(color=seg_colors[i], width=3),
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
                        line=dict(color='rgba(200,200,200,0.3)', width=1, dash='dot'),
                        showlegend=False, hoverinfo='skip',
                    ))

    fig.update_layout(
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                   aspectmode='data'),
        title=f'Horizon Evaluation (m={horizon}, {len(segments)} segments)',
        width=900, height=700,
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
                      yaxis_title='Error (cm)', xaxis_title='Step', height=400)
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
                      height=400)
    return fig


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def create_app(results_dir):
    results = load_results(results_dir)
    if not results:
        raise RuntimeError(f"No .npz files found in {results_dir}")

    episode_choices = sorted(results.keys())

    def update_plots(ep_idx, show_gt_line, show_segments, show_error_lines):
        ep_idx = int(ep_idx)
        r = results[ep_idx]
        fig_3d = build_3d_figure(r, show_gt_line, show_segments, show_error_lines)
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
        return fig_3d, fig_bar, fig_seg, summary

    with gr.Blocks(title="3DFA Horizon Evaluation") as app:
        gr.Markdown("# 3DFA Horizon Evaluation Viewer")
        gr.Markdown(
            "Interactive 3D visualization of GT-anchored autoregressive rollouts. "
            "Green diamonds = GT keyposes (anchors). Colored lines = predicted segments."
        )

        with gr.Row():
            ep_dropdown = gr.Dropdown(
                choices=[str(e) for e in episode_choices],
                value=str(episode_choices[0]), label="Episode",
            )
            show_gt = gr.Checkbox(value=True, label="Show GT path line")
            show_segs = gr.Checkbox(value=True, label="Show predicted segments")
            show_err = gr.Checkbox(value=False, label="Show error lines (pred->GT)")

        summary_md = gr.Markdown()

        with gr.Row():
            plot_3d = gr.Plot(label="3D Trajectory")
        with gr.Row():
            plot_bar = gr.Plot(label="Error vs Rollout Step")
            plot_seg = gr.Plot(label="Error per Segment")

        inputs = [ep_dropdown, show_gt, show_segs, show_err]
        outputs = [plot_3d, plot_bar, plot_seg, summary_md]

        ep_dropdown.change(update_plots, inputs, outputs)
        show_gt.change(update_plots, inputs, outputs)
        show_segs.change(update_plots, inputs, outputs)
        show_err.change(update_plots, inputs, outputs)
        app.load(update_plots, inputs, outputs)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="offline_evaluation/results")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = create_app(args.results_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
