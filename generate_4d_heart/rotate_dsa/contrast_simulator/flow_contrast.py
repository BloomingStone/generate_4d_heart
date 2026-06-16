from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import hashlib
import warnings
import logging

import numpy as np
import torch
from cupyx.scipy.ndimage import distance_transform_edt
import cupy as cp

from generate_4d_heart import CavityLabel, MU_IDODINE, MU_WATER, VesselLabel

from .contrast_simulator import ContrastSimulator

from skimage.morphology import skeletonize


def tensor_to_cupy(tensor: torch.Tensor) -> cp.ndarray:
    if tensor.device.type == 'cpu':
        return cp.asarray(tensor.numpy())
    else:
        return cp.from_dlpack(tensor)

def cupy_to_tensor(array: cp.ndarray, device: torch.device) -> torch.Tensor:
    if device.type == 'cpu':
        return torch.from_numpy(cp.asnumpy(array))
    else:
        return torch.from_dlpack(array)

@dataclass
class FlowContrast(ContrastSimulator):
    r"""Time-varying coronary contrast simulator.

    The start voxel is cached once because it is assumed to be anatomically stable.
    The coronary skeleton is recomputed on every call because the vessel can deform.
    The output density baseline follows the same HU-to-attenuation style as StaticIodineContrast.
    
    $$
    v = v0(r / r0) ^ \beta \\
    \tau = d / v \\
    x_{in} = \alpha (t - \tau - t_{in}) \\
    x_{out} = \alpha (t - \tau - t_{out}) \\
    \rho(d, t) = \mu_{water} + (\mu_{idodine} - \mu_{water}) \cdot (\sigma(x_{in}) - \sigma(x_{out}))
    $$
    v(velocity): flow velocity (voxels/s)
    t_in: time offset for input signal (s)
    t_out: time offset for output signal (s)
    alpha: controls the sharpness of the contrast change (higher alpha -> sharper change)
    """

    contrast_change_over_time: bool = True

    # Baseline density parameters, mirroring StaticIodineContrast's attenuation scale.
    mu_water_dsa: float = MU_WATER
    mu_idodine: float = MU_IDODINE

    # Time model parameters for rho(d, t)
    velocity: float = 200   # mm/s  v0
    t_in: float = 0.15
    t_out: float = 2.7
    alpha: float = 5.0
    beta: float = 1.5   # controls how velocity changes with vessel radius, v ~ (r/r0)^beta

    _cached_start_voxel: tuple[int, int, int] | None = field(default=None, init=False, repr=False)
    _cached_distance_map: torch.Tensor | None = field(default=None, init=False, repr=False)
    _cached_cor_hash: str | None = field(default=None, init=False, repr=False)   # if coronary label changes, cached skeleton and distance map should be invalidated

    device: torch.device = field(default=torch.device("cpu"), init=False, repr=False)

    def simulate(
        self,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        affine: np.ndarray,
    ) -> torch.Tensor:
        raise NotImplementedError("FlowContrast does not support simulate without time, please use `simulate_with_time` instead")

    def preprocess(
        self,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        affine: np.ndarray,
    ) -> torch.Tensor:
        """Return baseline attenuation map (no coronary iodine applied)."""
        density = ori_volume.clone().to(torch.float32)
        density = density / 1000.0 * self.mu_water_dsa + self.mu_water_dsa

        cavity = cavity_label.squeeze().to(torch.uint8)

        masked_volume = ori_volume.squeeze()[cavity == CavityLabel.LA]
        if masked_volume.numel() > 0:
            v_min = torch.quantile(masked_volume, 0.1 / 100)
            v_max = torch.quantile(masked_volume, 99.9 / 100)
            threshold_mask = (ori_volume.squeeze() > v_min) & (ori_volume.squeeze() < v_max)
            density.squeeze()[cavity > 0] = self.mu_water_dsa
            density.squeeze()[threshold_mask] = self.mu_water_dsa

        density.squeeze()[ori_volume.squeeze() < -2000] = 0.0
        
        # reinit cached values as the preprocess step is usually called before simulate_with_time, and the coronary label may change between different calls of preprocess.
        self._cached_start_voxel = None
        self._cached_distance_map = None
        self._cached_cor_hash = None
        
        return density

    def simulate_with_time(
        self,
        time: float,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        affine: np.ndarray,
    ) -> torch.Tensor:
        assert self.contrast_change_over_time is True, "FlowContrast only supports contrast change over time"
        assert ori_volume.ndim == 5 and ori_volume.shape[:2] == (1, 1), "ori_volume must be (1, 1, D, H, W)"
        assert cavity_label.shape == ori_volume.shape, "cavity_label must match ori_volume shape"
        assert coronary_label.shape == ori_volume.shape, "coronary_label must match ori_volume shape"
        assert coronary_label.dtype == torch.bool or coronary_label.dtype == torch.uint8, "coronary_label must be boolean or uint8"
        
        self.device = ori_volume.device
        cavity_label = cavity_label.to(self.device)
        coronary_label = coronary_label.to(self.device)

        if not coronary_label.any():
            raise ValueError("coronary_label is empty, cannot apply FlowContrast")
        
        coronary_label = coronary_label.squeeze()
        coronary_label = (coronary_label > 0).contiguous()
        
        # Check if coronary label has changed since last call, if so invalidate cached skeleton and distance map
        current_cor_hash = hashlib.md5(coronary_label.cpu().numpy().tobytes()).hexdigest()

        if current_cor_hash != self._cached_cor_hash:
            # Coronary label has changed, invalidate cached skeleton and distance map
            self._cached_distance_map = None
            self._cached_start_voxel = None
            self._cached_cor_hash = current_cor_hash

        # Compute distance map from coronary voxels to start voxel along the skeleton, then apply time-density model 
        # to get coronary density, and merge with baseline for output
        # Computed in the first time and cached for subsequent calls with the same coronary label.
        
        spacing = np.abs(np.diag(affine[:3, :3]))
        spacing = cp.asarray(spacing, dtype=cp.float32)
        
        if self._cached_distance_map is not None:
            distance_map = self._cached_distance_map.to(device=self.device)
        else:
            logging.debug("Computing skeleton and distance map...")
            skeleton_mask = self._compute_skeleton_with_radius(tensor_to_cupy(coronary_label), spacing)
            start_voxel = self._resolve_start_voxel(tensor_to_cupy(cavity_label.squeeze()), skeleton_mask)
            distance_map = self._compute_path_distance_map(skeleton_mask, start_voxel, spacing).to(device=self.device)
        
        density = ori_volume.squeeze().clone()
        coronary_density = self._density_from_distance(distance_map, float(time)).to(
            device=density.device,
            dtype=density.dtype,
        )
        density[coronary_label] = coronary_density[coronary_label]
        return density.unsqueeze(0).unsqueeze(0)

    def _resolve_start_voxel(self, cavity: cp.ndarray, skeleton_mask: cp.ndarray) -> tuple[int, int, int]:
        if self._cached_start_voxel is not None:
            return self._cached_start_voxel
        
        assert self._cached_start_voxel is None, "Start voxel should be None before first computation"

        if not skeleton_mask.any():
            raise ValueError("Unable to resolve a start voxel for FlowContrast: empty skeleton")

        aorta_mask = (cavity == int(VesselLabel.AORTA))
        
        if not aorta_mask.any():
            warnings.warn("There is no aorta label in the cavity, trying to find start voxel from LA/RA's center")
            la_ra = (cavity == int(CavityLabel.LA)) | (cavity == int(CavityLabel.RA))
            if not la_ra.any():
                raise ValueError("Unable to resolve a start voxel for FlowContrast: no aorta or LA/RA labels found")
            la_ra_center = cp.array(cp.argwhere(la_ra)).mean(axis=0).astype(int)
            start = self._nearest_skeleton_voxel(la_ra_center, skeleton_mask)
        else:
            start = self._nearest_skeleton_voxel_to_region(aorta_mask, skeleton_mask)

        self._cached_start_voxel = start
        return start

    @staticmethod
    def _compute_skeleton_with_radius(coronary_mask: cp.ndarray, spacing: cp.ndarray) -> cp.ndarray:
        """
        1. Compute the skeleton of the coronary mask.
        2. Compute the distance from each skeleton voxel to the nearest coronary boundary, and assign this distance to the skeleton voxel's value.
           This distance can be interpreted as the local vessel radius, and will be used to compute the Equivalent Flow Distance at each skeleton point.
        """
        
        if coronary_mask.ndim != 3:
            raise ValueError("coronary_label must be 3D after squeeze")

        # skeleton must be computed on CPU because skimage's skeletonize does not support GPU and cupyx doesn't have a built-in skeletonize function.
        skeleton = skeletonize(coronary_mask.get().astype(bool), method="lee").astype(bool)
        skeleton = cp.asarray(skeleton)
        
        if not skeleton.any():
            raise ValueError("Unable to compute skeleton for coronary mask: empty skeleton")
        
        # Add a radius to the skeleton based on the distance to the coronary boundary
        distance_to_boundary = distance_transform_edt(coronary_mask, sampling=spacing.tolist())  # convert to mm  #type: ignore
        res = cp.zeros_like(skeleton, dtype=float)
        assert distance_to_boundary is not None, "distance_transform_edt should return a distance map when input is not empty"
        res[skeleton] = distance_to_boundary[skeleton]
        
        return res

    def _compute_path_distance_map(self, skeleton_mask_with_radius: cp.ndarray, start_voxel: tuple[int, int, int], spacing: cp.ndarray) -> torch.Tensor:
        coords = cp.argwhere(skeleton_mask_with_radius>0)
        r0 = skeleton_mask_with_radius[skeleton_mask_with_radius>0].max()  # minimum radius along the skeleton, used as reference radius for flow model
        if len(coords) == 0:
            raise ValueError("Empty skeleton")
        
        # map coord tuple -> index
        coord_to_index = {self._coord_tuple(coord): idx for idx, coord in enumerate(coords)}
        start_node = self._nearest_skeleton_voxel(start_voxel, skeleton_mask_with_radius)
        start_idx = coord_to_index.get(start_node)
        if start_idx is None:
            # fallback: pick nearest skeleton coord
            start_idx = 0

        # Dijkstra on skeleton graph (26-neighborhood)
        n = len(coords)
        distances_voxel = cp.full(n, cp.inf, dtype=cp.float32)
        distances_voxel[start_idx] = 0.0
        distances_world = cp.full(n, cp.inf, dtype=cp.float32)
        distances_world[start_idx] = 0.0
        
        queue: list[tuple[float, float, int]] = [(0.0, 0.0, start_idx)]
        while queue:
            d_voxel, d_world, idx = heapq.heappop(queue)
            if d_voxel > distances_voxel[idx]:
                continue
            x, y, z = self._coord_tuple(coords[idx])
            for offset in self._neighbor_offsets():
                neighbor = (x + offset[0], y + offset[1], z + offset[2])
                neighbor_idx = coord_to_index.get(neighbor)
                if neighbor_idx is None:
                    continue
                offset_world = cp.asarray(offset, dtype=cp.float32) * cp.asarray(spacing, dtype=cp.float32)
                step_world = float(cp.linalg.norm(cp.asarray(offset_world, dtype=cp.float32)))
                step_voxel = float(cp.linalg.norm(cp.asarray(offset, dtype=cp.float32)))
                
                radius = skeleton_mask_with_radius[x, y, z]  # the radius at current skeleton point
                
                # v = v0 * (r / r0)^\beta  (Poiseuille's law)
                v = self.velocity * (radius / r0)**self.beta if radius > 0 else self.velocity
                
                new_d_world = d_world + step_world / max(v, 1e-6)
                new_d_voxel = d_voxel + step_voxel
                if new_d_voxel < distances_voxel[neighbor_idx]:
                    distances_voxel[neighbor_idx] = new_d_voxel
                    distances_world[neighbor_idx] = new_d_world
                    heapq.heappush(queue, (new_d_voxel, new_d_world, neighbor_idx))

        skel_dist = cp.zeros(skeleton_mask_with_radius.shape, dtype=cp.float32)
        for idx, coord in enumerate(coords):
            skel_dist[self._coord_tuple(coord)] = distances_world[idx]

        # Project skeleton distances back to full volume via nearest-skeleton assignment
        _, nearest_indices  = distance_transform_edt(~(skeleton_mask_with_radius>0), return_distances=True, return_indices=True)  # type: ignore
        assert nearest_indices is not None, "distance_transform_edt with return_indices=True should return indices"
        projected = skel_dist[nearest_indices[0], nearest_indices[1], nearest_indices[2]]

        # cache and return
        self._cached_distance_map = cupy_to_tensor(projected, device=self.device)
        return self._cached_distance_map

    def _density_from_distance(self, distance_map: torch.Tensor, time: float) -> torch.Tensor:
        tau = distance_map
        x_in = self.alpha * (time - tau - self.t_in)
        x_out = self.alpha * (time - tau - self.t_out)
        pulse = torch.sigmoid(x_in) - torch.sigmoid(x_out)
        return (self.mu_water_dsa + (self.mu_idodine - self.mu_water_dsa) * pulse).to(torch.float32)

    def _nearest_skeleton_voxel(self, start_voxel: tuple[int, int, int], skeleton_mask: cp.ndarray) -> tuple[int, int, int]:
        if skeleton_mask[start_voxel].all():
            return start_voxel
        coords = cp.argwhere(skeleton_mask>0)
        if len(coords) == 0:
            return start_voxel
        distances = cp.linalg.norm(coords.astype(cp.float32) - cp.asarray(start_voxel, dtype=cp.float32), axis=1)
        nearest_coord = coords[int(cp.argmin(distances))]
        return self._coord_tuple(nearest_coord)

    def _nearest_skeleton_voxel_to_region(self, region_mask: cp.ndarray, skeleton_mask: cp.ndarray) -> tuple[int, int, int]:
        coords = cp.argwhere(skeleton_mask > 0)
        if len(coords) == 0:
            raise ValueError("Unable to find a skeleton voxel")
        region_coords = cp.argwhere(region_mask)
        if len(region_coords) == 0:
            nearest = coords[0]
        else:
            # Use cupyx KDTree to avoid O(N*M) GPU memory blowup from pairwise distances,
            # while keeping everything on GPU.
            from cupyx.scipy.spatial import KDTree
            tree = KDTree(coords.astype(cp.float32))
            dists, nearest_skel_idxs = tree.query(region_coords.astype(cp.float32), k=1)
            best_region_idx = int(cp.argmin(dists))
            nearest = coords[nearest_skel_idxs[best_region_idx]]
        return self._coord_tuple(nearest)

    @staticmethod
    def _coord_tuple(coord: cp.ndarray) -> tuple[int, int, int]:
        return int(coord[0]), int(coord[1]), int(coord[2])

    @staticmethod
    def _neighbor_offsets() -> list[tuple[int, int, int]]:
        offsets: list[tuple[int, int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    offsets.append((dx, dy, dz))
        return offsets
