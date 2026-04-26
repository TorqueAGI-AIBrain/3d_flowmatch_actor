# episode_viewer.py: Load and visualize a single extracted episode with Open3D.
# episode_viewer.py: Point clouds from depth+RGB+extrinsics, EEF trajectory overlay.

"""
Dataset class for extracted episode directories. Each instance loads one
episode and provides methods to access frames and visualize point clouds
with Open3D.

Handles both static (ncam, 4, 4) and per-frame (T, ncam, 4, 4) extrinsics.

Usage:
    python -m data_processing.episode_viewer \
        --episode_dir data/xarm/default_task_fixed_v3_episodes/default_task_fixed_v3/episode_0 \
        --frame 50

    # Visualize multiple frames
    python -m data_processing.episode_viewer \
        --episode_dir ... --frame 0 --frame 50 --frame 100
"""

import argparse
import os

import cv2
import numpy as np


# Episode depth PNGs are uint16 millimetres (same constant as bag_to_episodes.py)
DEPTH_MM_SCALE = 1000.0


class EpisodeDataset:
    """Load and access a single extracted episode."""

    def __init__(self, episode_dir, cam_names=None):
        self.episode_dir = episode_dir

        # Load camera parameters
        self.extrinsics = np.load(os.path.join(episode_dir, 'camera_extrinsics.npy'))
        self.intrinsics = np.load(os.path.join(episode_dir, 'camera_intrinsics.npy'))
        self.eef_states = np.load(os.path.join(episode_dir, 'eef_states.npy'))

        # Detect per-frame vs static extrinsics
        # Per-frame: (T, ncam, 4, 4), static: (ncam, 4, 4)
        if self.extrinsics.ndim == 4:
            self.per_frame_extrinsics = True
            self.ncam = self.extrinsics.shape[1]
        else:
            self.per_frame_extrinsics = False
            self.ncam = self.extrinsics.shape[0]

        self.T = len(self.eef_states)

        # Detect camera names from files in rgb/
        if cam_names is not None:
            self.cam_names = cam_names
        else:
            rgb_dir = os.path.join(episode_dir, 'rgb')
            names = set()
            for f in os.listdir(rgb_dir):
                if f.endswith('.png'):
                    # front_0000.png -> front
                    name = '_'.join(f.replace('.png', '').split('_')[:-1])
                    names.add(name)
            self.cam_names = sorted(names)

        print(f"Episode: {episode_dir}")
        print(f"  Frames: {self.T}, Cameras: {self.cam_names}")
        print(f"  Extrinsics: {self.extrinsics.shape} "
              f"({'per-frame' if self.per_frame_extrinsics else 'static'})")
        print(f"  Intrinsics: {self.intrinsics.shape}")

    def get_extrinsic(self, cam_idx, frame_idx=0):
        """Get 4x4 extrinsic for a camera at a frame."""
        if self.per_frame_extrinsics:
            return self.extrinsics[frame_idx, cam_idx].astype(np.float64)
        return self.extrinsics[cam_idx].astype(np.float64)

    def get_intrinsic(self, cam_idx):
        """Get 3x3 intrinsic for a camera."""
        return self.intrinsics[cam_idx].astype(np.float64)

    def get_rgb(self, cam_idx, frame_idx):
        """Load RGB image (H, W, 3) uint8."""
        path = os.path.join(self.episode_dir, 'rgb',
                            f'{self.cam_names[cam_idx]}_{frame_idx:04d}.png')
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_depth(self, cam_idx, frame_idx):
        """Load depth map (H, W) in metres (float32)."""
        path = os.path.join(self.episode_dir, 'depth',
                            f'{self.cam_names[cam_idx]}_{frame_idx:04d}.png')
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # uint16
        return raw.astype(np.float32) / DEPTH_MM_SCALE

    def get_eef_pose(self, frame_idx):
        """Get EEF state [x, y, z, qx, qy, qz, qw, gripper]."""
        return self.eef_states[frame_idx]

    def unproject(self, cam_idx, frame_idx, depth_min=0.01, depth_max=3.0):
        """Unproject depth to world-frame point cloud.

        Returns:
            pts: (N, 3) float64 points in world frame
            colors: (N, 3) float64 RGB [0, 1]
        """
        rgb = self.get_rgb(cam_idx, frame_idx)
        depth = self.get_depth(cam_idx, frame_idx)
        K = self.get_intrinsic(cam_idx)
        E = self.get_extrinsic(cam_idx, frame_idx)

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                           np.arange(h, dtype=np.float64))
        z = depth.astype(np.float64).flatten()
        x = (u.flatten() - K[0, 2]) * z / K[0, 0]
        y = (v.flatten() - K[1, 2]) * z / K[1, 1]

        pts_cam = np.stack([x, y, z], axis=1)
        valid = (z > depth_min) & (z < depth_max) & np.isfinite(pts_cam).all(1)

        pts_cam = pts_cam[valid]
        colors = rgb.reshape(-1, 3)[valid].astype(np.float64) / 255.0

        # Transform to world frame
        pts_world = (E[:3, :3] @ pts_cam.T).T + E[:3, 3]

        return pts_world, colors

    def visualize_frame(self, frame_idx, show_trajectory=True,
                        show_cameras=None):
        """Visualize point clouds from all cameras at a frame using Open3D.

        Args:
            frame_idx: which frame to visualize
            show_trajectory: overlay EEF trajectory as a line
            show_cameras: list of camera indices, or None for all
        """
        import open3d as o3d

        geometries = []

        if show_cameras is None:
            show_cameras = list(range(self.ncam))

        # Point clouds per camera
        cam_colors_lut = [
            None,  # use real RGB
            None,
        ]
        for cam_idx in show_cameras:
            pts, colors = self.unproject(cam_idx, frame_idx)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(pcd)

            # Camera position marker
            E = self.get_extrinsic(cam_idx, frame_idx)
            cam_pos = E[:3, 3]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(cam_pos)
            sphere.paint_uniform_color([1.0, 0.0, 0.0] if cam_idx == 0 else [0.0, 0.0, 1.0])
            geometries.append(sphere)

        # Coordinate frame at link_base origin
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        geometries.append(axes)

        # EEF trajectory
        if show_trajectory:
            eef_positions = self.eef_states[:, :3].astype(np.float64)
            # Full trajectory as line
            lines = [[i, i + 1] for i in range(len(eef_positions) - 1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(eef_positions)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0.0, 1.0, 0.0])
            geometries.append(line_set)

            # Current EEF position as sphere
            eef_pos = eef_positions[frame_idx]
            eef_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            eef_sphere.translate(eef_pos)
            eef_sphere.paint_uniform_color([1.0, 1.0, 0.0])
            geometries.append(eef_sphere)

        print(f"\nFrame {frame_idx}/{self.T}:")
        for cam_idx in show_cameras:
            E = self.get_extrinsic(cam_idx, frame_idx)
            print(f"  {self.cam_names[cam_idx]}: t=[{E[0,3]:.4f}, {E[1,3]:.4f}, {E[2,3]:.4f}]")
        eef = self.get_eef_pose(frame_idx)
        print(f"  EEF: [{eef[0]:.4f}, {eef[1]:.4f}, {eef[2]:.4f}]")

        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Episode Frame {frame_idx}",
            width=1280, height=720,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_dir', type=str, required=True)
    parser.add_argument('--frame', type=int, action='append', default=None,
                        help='Frame index to visualize (can repeat)')
    parser.add_argument('--cam_names', type=str, default=None,
                        help='Comma-separated camera names (auto-detected if omitted)')
    args = parser.parse_args()

    cam_names = args.cam_names.split(',') if args.cam_names else None
    ds = EpisodeDataset(args.episode_dir, cam_names=cam_names)

    frames = args.frame if args.frame else [0]
    for f in frames:
        if f < 0:
            f = ds.T + f
        ds.visualize_frame(f)


if __name__ == '__main__':
    main()
