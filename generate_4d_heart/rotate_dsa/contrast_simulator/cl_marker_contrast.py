from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import product

from typing import override
import torch
import numpy as np
from skimage.morphology import skeletonize

from .contrast_simulator import ContrastSimulator
from generate_4d_heart import CavityLabel, MU_WATER, MU_IDODINE, MU_GUIDE_WIRE, MU_MARKER, VesselLabel


# TODO 与 Artifacs 重复，需要删除
@dataclass
class ClMarkerSimulator(ContrastSimulator):
    """
    冠脉造影 + 导丝 + 标记点 模拟器。

    模拟流程:
    1. 将原始 HU 图像转换为衰减图，腔体区域归一化到水基线
    2. 冠脉区域填充碘（或水）
    3. 从主动脉出发沿冠脉骨架随机选一条路径作为导丝（如果未从外部给定）
    4. 在导丝上随机采样两个标记点，以球体形式写入，带距离衰减（如果未从外部给定）
    """
    mu_idodine: float = MU_IDODINE
    mu_water: float = MU_WATER
    mu_guide_wire: float = MU_GUIDE_WIRE
    mu_marker: float = MU_MARKER
    
    has_idodine: bool = False
    """ 是否在冠脉区域使用碘衰减值，False 则使用水衰减值。"""
    
    margin_ratio: float = 0.9
    """ markers为None时, 决定随机标记点采样时的高斯概率分布边界
    （3 \\sigma = margin_ratio * max(len(AABB)))，。
    设置该值的目的是为了防止随机采样的marker点靠近导丝的边界
    """
    
    marker_distance : float = 6    #mm
    """ markers为None时, 随机采样的markers的间距， 单位 mm """
    
    marker_radius: int = 1
    """ 单个marker的球体半径，单位体素。衰减值在球体内随距离中心点的距离衰减。"""

    contrast_change_over_time: bool = field(default=False, init=False)
    
    guide_line: np.ndarray | None = None
    """ given guide line, if None, will be computed from coronary and cavity labels """
    
    markers: list[tuple[int, int, int]]|None = None
    """ given markers, if None, will be computed from guide line """
    
    @override
    def preprocess(
        self,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        affine: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert HU image to baseline attenuation map and normalize cavity/threshold regions to water baseline.
        """
        res = ori_volume.clone()
        res = res / 1000.0 * self.mu_water + self.mu_water  # HU -> attenuation

        assert cavity_label.dtype == torch.uint8

        # Use left atrium region to estimate typical enhanced range
        masked_volume = ori_volume.squeeze()[cavity_label.squeeze() == CavityLabel.LA]
        if masked_volume.numel() > 0:
            v_min = torch.quantile(masked_volume, 0.1 / 100)
            v_max = torch.quantile(masked_volume, 99.9 / 100)
            threshold_mask = (ori_volume > v_min) & (ori_volume < v_max)
            res[(cavity_label > 0) | threshold_mask] = self.mu_water

        # Invalid HU values -> zero attenuation
        res[ori_volume < -2000] = 0.0
        
        return res
    
    
        
    @override
    def simulate_with_time(
        self,
        time: float, 
        ori_volume: torch.Tensor, 
        cavity_label: torch.Tensor, 
        coronary_label: torch.Tensor, 
        affine: np.ndarray
    ):
        raise NotImplementedError
    
    @override
    def simulate(
        self, 
        ori_volume: torch.Tensor, 
        cavity_label: torch.Tensor, 
        coronary_label: torch.Tensor, 
        affine: np.ndarray
    ):
        """
        在预处理后的衰减图上叠加冠脉造影、导丝和标记点。

        参数
        ----------
        ori_volume : torch.Tensor
            预处理后的基线衰减图 (preprocess 输出)。
        cavity_label : torch.Tensor
            腔体标签，dtype=uint8。
        coronary_label : torch.Tensor
            冠脉二值 mask。
        affine : np.ndarray
            4x4 仿射变换矩阵。

        返回值
        -------
        torch.Tensor
            叠加了冠脉碘造影、导丝路径和标记点衰减的模拟 DSA 图像。
        """
        # `ori_volume` is expected to be a preprocessed attenuation-like baseline
        res = ori_volume.clone().squeeze()
        assert cavity_label.dtype == torch.uint8
        assert coronary_label.dtype == torch.bool or coronary_label.dtype == torch.uint8
        coronary_np = coronary_label.cpu().numpy().squeeze().astype(bool)
        cavity_np = cavity_label.cpu().numpy().squeeze()

        mu_coronary = self.mu_idodine if self.has_idodine else self.mu_water
        res[coronary_np > 0] = mu_coronary
        
        guide_line = self._get_guide_line(coronary_np, cavity_np) if self.guide_line is None else self.guide_line
        
        res[guide_line > 0] = self.mu_guide_wire
        
        markers = self._get_markers_from_guide_line(affine, guide_line) if self.markers is None else self.markers

        
        # Set small spheres centered at marker positions to mu_marker
        for marker in markers:
            mx, my, mz = marker
            radius = self.marker_radius  # sphere radius in voxels
            # Bounds checking
            x_min = max(0, mx - radius)
            x_max = min(res.shape[-3], mx + radius + 1)
            y_min = max(0, my - radius)
            y_max = min(res.shape[-2], my + radius + 1)
            z_min = max(0, mz - radius)
            z_max = min(res.shape[-1], mz + radius + 1)
            
            # Create sphere mask using meshgrid
            x = torch.arange(x_min, x_max, device=res.device)
            y = torch.arange(y_min, y_max, device=res.device)
            z = torch.arange(z_min, z_max, device=res.device)
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            dist2 = (xx - mx) ** 2 + (yy - my) ** 2 + (zz - mz) ** 2
            sphere_mask = dist2 <= radius ** 2
            
            res[x_min:x_max, y_min:y_max, z_min:z_max][sphere_mask] = self.mu_marker
        
        res = res.reshape(*ori_volume.shape)
        return res

    def _get_guide_line(self, coronary_label: np.ndarray, cavity_label: np.ndarray) -> np.ndarray:
        """
        Build a tree from start_point along the skeleton, and select the
        longest path to a leaf node as the guide line.
        """
        skeleton = skeletonize(coronary_label, method="lee").astype(bool)
        start_point = self._resolve_start_voxel(cavity_label, skeleton)

        # Get coordinates of all skeleton voxels
        coords = np.array(np.where(skeleton)).T  # (N, 3)
        coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}

        # Build 26-neighbor adjacency graph
        n = len(coords)
        adj = [[] for _ in range(n)]
        for i in range(n):
            ci = coords[i]
            for di, dj, dk in product((-1, 0, 1), repeat=3):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                nb_key = (int(ci[0] + di), int(ci[1] + dj), int(ci[2] + dk))
                if nb_key in coord_to_idx:
                    j = coord_to_idx[nb_key]
                    if j > i:
                        adj[i].append(j)
                        adj[j].append(i)

        # BFS from start_point to build a tree (parent pointers) with depth
        start_idx = coord_to_idx[tuple(start_point)]
        parent = np.full(n, -1, dtype=int)
        depth = np.full(n, -1, dtype=int)
        parent[start_idx] = start_idx
        depth[start_idx] = 0
        q: deque[int] = deque([start_idx])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if parent[nb] == -1:
                    parent[nb] = cur
                    depth[nb] = depth[cur] + 1
                    q.append(nb)

        # Find leaf nodes (nodes with no children in the BFS tree)
        has_child = np.zeros(n, dtype=bool)
        for i in range(n):
            p = parent[i]
            if p != -1 and p != i:
                has_child[p] = True
        leaves = [i for i in range(n) if parent[i] != -1 and not has_child[i]]

        if not leaves:
            # Fallback: return full skeleton
            guide_line = skeleton.copy()
            return guide_line

        # Select the longest path (deepest leaf)
        chosen_leaf = int(leaves[np.argmax([depth[l] for l in leaves])])
        path_indices = []
        cur = chosen_leaf
        while cur != start_idx:
            path_indices.append(cur)
            cur = int(parent[cur])
        path_indices.append(start_idx)

        # Build guide_line mask
        guide_line = np.zeros_like(skeleton, dtype=bool)
        for idx in path_indices:
            c = coords[idx]
            guide_line[c[0], c[1], c[2]] = True

        return guide_line
        
    
    def _get_markers_from_guide_line(self, affine: np.ndarray, sklt: np.ndarray) -> list[tuple[int, int, int]]:
        """
        从 guide_line 上随机提取两个点作为 marker, 提取流程如下:
        1. 计算 guide_line 的几何中心与 AABB, 取 AABB 最长边 max_aabb
        2. 以几何中心为均值、margin_ratio * max_aabb / 6 为 sigma 构造高斯采样概率，
           超出 margin_ratio * max_aabb / 2 的区域概率置零
        3. guide_line 外的点天然概率为零（coords 仅含 guide_line 点）
        4. 按概率随机采样作为点 1
        5. 以物理距离 (mm) 为边权重建 guide_line 邻接图，从 point1 出发 BFS，
           累积距离超过 self.marker_distance 时停止，将上一个节点作为 point 2
           （失败则仅返回点 1）
        """
        # Get coordinates of all skeleton voxels
        coords = np.array(np.where(sklt)).T  # (N, 3)
        
        # 1. Compute geometric center and AABB
        center = coords.mean(axis=0)
        aabb_min = coords.min(axis=0)
        aabb_max = coords.max(axis=0)
        aabb_size = aabb_max - aabb_min
        max_aabb = aabb_size.max()
        
        # 2. Gaussian sampling probability centered at geometric center
        sigma = self.margin_ratio * max_aabb / 6.0  # 3 sigma at margin_ratio*min_aabb/2
        margin_dist = self.margin_ratio * max_aabb / 2.0
        dists_from_center = np.linalg.norm(coords - center, axis=1)
        probs = np.exp(-0.5 * (dists_from_center / sigma) ** 2)
        # Zero probability beyond margin
        probs[dists_from_center > margin_dist] = 0.0
        # 3. Already masked by sklt since coords only contains skeleton points
        
        # 4. Randomly sample point 1
        probs = probs / probs.sum()
        idx1 = np.random.choice(len(coords), p=probs)
        point1 = coords[idx1]

        voxel_spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

        # 这里重新计算了一次 adj,而不用 get_guilde 中已经计算过的 是为了便于处理外部输入guide line
        # 所有计算仅在开始时运行一次，代价不大，且保证了代码的独立性
        # Precompute physical distance for each 26-neighbor offset
        offset_dist: dict[tuple[int, int, int], float] = {}
        for di, dj, dk in product((-1, 0, 1), repeat=3):
            if di == 0 and dj == 0 and dk == 0:
                continue
            offset_dist[(di, dj, dk)] = float(np.sqrt(
                (di * voxel_spacing[0]) ** 2 +
                (dj * voxel_spacing[1]) ** 2 +
                (dk * voxel_spacing[2]) ** 2
            ))

        coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
        n_gl = len(coords)
        
        adj_gl: list[list[tuple[int, float]]] = [[] for _ in range(n_gl)]
        for i in range(n_gl):
            ci = coords[i]
            for (di, dj, dk), d in offset_dist.items():
                nb_key = (int(ci[0] + di), int(ci[1] + dj), int(ci[2] + dk))
                j = coord_to_idx.get(nb_key)
                if j is not None and j > i:
                    adj_gl[i].append((j, d))
                    adj_gl[j].append((i, d))

        # Locate point1 index
        p1_idx = coord_to_idx.get(tuple(point1))
        
        # If point1 is not found in the adjacency graph (should not happen), find the nearest neighbor
        if p1_idx is None:
            p1_idx = int(np.argmin(np.linalg.norm(
                coords.astype(np.float64) - point1.astype(np.float64), axis=1
            )))

        # BFS from point1: stop when cumulative distance > marker_distance
        visited = np.zeros(n_gl, dtype=bool)
        visited[p1_idx] = True
        dist_from_p1 = np.full(n_gl, np.inf, dtype=np.float64)
        dist_from_p1[p1_idx] = 0.0
        q: deque[int] = deque([p1_idx])

        while q:
            cur = q.popleft()
            for nb, d in adj_gl[cur]:
                if visited[nb]:
                    continue
                visited[nb] = True
                new_dist = dist_from_p1[cur] + d
                dist_from_p1[nb] = new_dist
                if new_dist > self.marker_distance:
                    # cur is the last node before exceeding marker_distance
                    if cur == p1_idx:
                        print("Warning: Single edge exceeds marker_distance. Returning only point1.")
                        return [tuple(point1.astype(int)), ]
                    point2 = coords[cur]
                    return [tuple(point1.astype(int)), tuple(point2.astype(int))]
                q.append(nb)

        # Path exhausted without reaching marker_distance
        print("Warning: Guide line too short for marker_distance. Returning only point1.")
        return [tuple(point1.astype(int)), ]
    
    def _resolve_start_voxel(self, cavity: np.ndarray, skeleton_mask: np.ndarray) -> tuple[int, int, int]:
        if not skeleton_mask.any():
            raise ValueError("Unable to resolve a start voxel for cl_marker_contrast: empty skeleton")

        aorta_mask = (cavity == int(VesselLabel.AORTA))
        
        if not aorta_mask.any():
            raise ValueError("Unable to resolve a start voxel for cl_marker_contrast: empty aorta region")
        
        start = self._nearest_skeleton_voxel_to_region(aorta_mask, skeleton_mask)

        return start
    
    def _nearest_skeleton_voxel_to_region(self, region_mask: np.ndarray, skeleton_mask: np.ndarray) -> tuple[int, int, int]:
        coords = np.argwhere(skeleton_mask > 0)
        if len(coords) == 0:
            raise ValueError("Unable to find a skeleton voxel")
        
        region_coords = np.argwhere(region_mask)
        if len(region_coords) == 0:
            return self._coord_tuple(coords[0])

        # Use KDTree to avoid O(N*M) GPU memory blowup from pairwise distances,
        # while keeping everything on GPU.
        from scipy.spatial import KDTree
        tree = KDTree(coords.astype(np.float32))
        dists, nearest_skel_idxs = tree.query(region_coords.astype(np.float32), k=1)
        best_region_idx = int(np.argmin(dists))
        nearest = coords[nearest_skel_idxs[best_region_idx]]
        return self._coord_tuple(nearest)
    
    @staticmethod
    def _coord_tuple(coord: np.ndarray) -> tuple[int, int, int]:
        return int(coord[0]), int(coord[1]), int(coord[2])