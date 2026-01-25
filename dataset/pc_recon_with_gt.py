import os
import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d


class PCReconWithGT(Dataset):
    """
    读取真实点云与匹配的 GT 网格（按同名 .ply 配对），并在同一中心/尺度下归一化两者。
    - 点云位置将按 delta 体素降采样（与 PCReconSet 一致，可选 rescale_delta）。
    - 返回 (points[N,3] Tensor, gt_vertices[M,3] Tensor)，两者均已基于点云的 center/scale 归一化。
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

        # 构建同名 .ply 的 GT 路径，若不存在则跳过该样本
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

        # 读点云并可选体素下采样
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

        # 以点云的包围盒中心/尺度为基准做归一化
        center, scale = self._center_scale(pc_points)
        pc_points_norm = (pc_points - center) / scale

        # 读取 GT（网格或点云）并用相同的 center/scale 归一化
        # 某些数据（如 CARLA_1M）GT 可能是无面的点云 .ply，直接按点云读取可避免 Open3D 的 TriangleMesh 警告
        def _ply_has_faces(p: str) -> bool:
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        if i > 100:  # 仅检查头部若干行
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
            # 无面：按点云读取
            gt_pc = o3d.io.read_point_cloud(gt_path)
            gt_vertices = np.asarray(gt_pc.points, dtype=np.float32)
        else:
            # 有面或非 ply：按三角网格读取
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
