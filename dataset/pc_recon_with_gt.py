import os
import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d


class PCReconWithGT(Dataset):
    """
    Load real point clouds and their matching GT meshes (paired by same-name .ply),
    and normalize both to the same center/scale.
    - Point cloud positions will be voxel-downsampled by delta (same behavior as PCReconSet; optional rescale_delta).
    - Returns (points[N,3] Tensor, gt_vertices[M,3] Tensor), both normalized using the point cloud's center/scale.
    """

    def __init__(self, data_root: str, dataset: str, delta: float = 0.01, rescale_delta: bool = False):
        super().__init__()
        self.data_root = data_root
        self.dataset = dataset
        self.delta = float(delta)
        self.rescale_delta = bool(rescale_delta)

        pc_dir = os.path.join(self.data_root, 'PointClouds', self.dataset)
        gt_dir = os.path.join(self.data_root, 'GT_Meshes', self.dataset)
        self.pcloud_files: List[str] = sorted(glob.glob(os.path.join(pc_dir, '*.ply')))

        # Build matching GT .ply paths; skip samples if GT is missing
        pairs: List[Tuple[str, str]] = []
        for pc_path in self.pcloud_files:
            stem = os.path.splitext(os.path.basename(pc_path))[0]
            gt_path = os.path.join(gt_dir, f"{stem}.ply")
            if os.path.isfile(gt_path):
                pairs.append((pc_path, gt_path))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _center_scale(points: np.ndarray):
        min_bnd = np.min(points, axis=0)
        max_bnd = np.max(points, axis=0)
        center = (min_bnd + max_bnd) / 2
        scale = np.linalg.norm(max_bnd - min_bnd) / 2
        return center, scale

    def __getitem__(self, idx: int):
        pc_path, gt_path = self.pairs[idx]

        # Read point cloud and optionally voxel-downsample
        pcd = o3d.io.read_point_cloud(pc_path)
        pcd = pcd.remove_duplicated_points()
        if self.delta == 0:
            pc_points = np.asarray(pcd.points)
        else:
            if self.rescale_delta:
                X = np.asarray(pcd.points)
                scale = self._center_scale(X)[-1]
                voxel = self.delta * scale
            else:
                voxel = self.delta
            voxel_pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
            pc_points = np.asarray(voxel_pcd.points)

        # Normalize based on the point cloud's bounding box center/scale
        center, scale = self._center_scale(pc_points)
        pc_points_norm = (pc_points - center) / scale

        # Read GT (mesh or point cloud) and normalize with the same center/scale
        # Some GTs (e.g., CARLA_1M) may be .ply point clouds without faces;
        # reading them as point clouds avoids Open3D TriangleMesh warnings
        def _ply_has_faces(p: str) -> bool:
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        if i > 100:  # only check the first few header lines
                            break
                        if line.startswith('element face'):
                            parts = line.strip().split()
                            if len(parts) >= 3 and parts[2].isdigit():
                                return int(parts[2]) > 0
            except Exception:
                pass
            return False

        gt_vertices: np.ndarray
        if gt_path.lower().endswith('.ply') and not _ply_has_faces(gt_path):
            # No faces: read as point cloud
            gt_pc = o3d.io.read_point_cloud(gt_path)
            gt_vertices = np.asarray(gt_pc.points, dtype=np.float32)
        else:
            # Has faces or non-ply: read as triangle mesh
            mesh = o3d.io.read_triangle_mesh(gt_path)
            try:
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
            except Exception:
                pass
            gt_vertices = np.asarray(mesh.vertices, dtype=np.float32)
        gt_vertices_norm = (gt_vertices - center) / scale

        points_t = torch.from_numpy(pc_points_norm.astype('float32'))
        gt_t = torch.from_numpy(gt_vertices_norm.astype('float32'))
        return points_t, gt_t
