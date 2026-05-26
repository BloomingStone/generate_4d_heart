from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import hashlib
import warnings

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from generate_4d_heart import CavityLabel, MU_IDODINE, MU_WATER, VesselLabel

from .contrast_simulator import ContrastSimulator

from skimage.morphology import skeletonize


@dataclass
class FlowContrast(ContrastSimulator):
    r"""Time-varying coronary contrast simulator.

    The start voxel is cached once because it is assumed to be anatomically stable.
    The coronary skeleton is recomputed on every call because the vessel can deform.
    The output density baseline follows the same HU-to-attenuation style as MultipliContrast.
    
    $$
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

    # Baseline density parameters, mirroring MultipliContrast's attenuation scale.
    mu_water_dsa: float = MU_WATER
    mu_idodine: float = MU_IDODINE

    # Time model parameters for rho(d, t)
    velocity: float = 400   # TODO 统一单位到mm/s
    t_in: float = 0.15
    t_out: float = 2.7
    alpha: float = 5.0

    _cached_start_voxel: tuple[int, int, int] | None = field(default=None, init=False, repr=False)
    _cached_distance_map: torch.Tensor | None = field(default=None, init=False, repr=False)
    _cached_cor_hash: str | None = field(default=None, init=False, repr=False)   # if coronary label changes, cached skeleton and distance map should be invalidated


    def simulate(
        self,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("FlowContrast does not support simulate without time, please use `simulate_with_time` instead")

    def preprocess(
        self,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
    ) -> torch.Tensor:
        """Return baseline attenuation map (no coronary iodine applied)."""
        return self._baseline_map(ori_volume, cavity_label)

    def _baseline_map(self, ori_volume: torch.Tensor, cavity_label: torch.Tensor) -> torch.Tensor:
        """
        Compute baseline attenuation from original HU and cavity labels (no coronary iodine).
        Returns a tensor shaped (1,1,D,H,W).
        """
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
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        assert self.contrast_change_over_time is True, "FlowContrast only supports contrast change over time"
        assert ori_volume.ndim == 5 and ori_volume.shape[:2] == (1, 1), "ori_volume must be (1, 1, D, H, W)"
        assert cavity_label.shape == ori_volume.shape, "cavity_label must match ori_volume shape"
        assert coronary_label.shape == ori_volume.shape, "coronary_label must match ori_volume shape"
        assert coronary_label.dtype == torch.bool or coronary_label.dtype == torch.uint8, "coronary_label must be boolean or uint8"
        
        cavity_label = cavity_label.to(ori_volume.device)
        coronary_label = coronary_label.to(ori_volume.device)

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
        if self._cached_distance_map is not None:
            distance_map = self._cached_distance_map.to(device=ori_volume.device)
        else:
            print("Computing skeleton and distance map for the first time...")
            skeleton_mask = self._compute_skeleton(coronary_label.cpu().numpy())
            start_voxel = self._resolve_start_voxel(cavity_label.squeeze().cpu().numpy(), skeleton_mask)
            distance_map = self._compute_path_distance_map(skeleton_mask, start_voxel).to(device=ori_volume.device)
        
        density = ori_volume.squeeze().clone()
        coronary_density = self._density_from_distance(distance_map, float(time)).to(
            device=density.device,
            dtype=density.dtype,
        )
        density[coronary_label] = coronary_density[coronary_label]
        return density.unsqueeze(0).unsqueeze(0)

    def _resolve_start_voxel(self, cavity: np.ndarray, skeleton_mask: np.ndarray) -> tuple[int, int, int]:
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
            la_ra_center = np.array(np.argwhere(la_ra)).mean(axis=0).astype(int)
            start = self._nearest_skeleton_voxel(la_ra_center, skeleton_mask)
        else:
            start = self._nearest_skeleton_voxel_to_region(aorta_mask, skeleton_mask)

        self._cached_start_voxel = start
        return start

    def _compute_skeleton(self, coronary_mask: np.ndarray) -> np.ndarray:
        if coronary_mask.ndim != 3:
            raise ValueError("coronary_label must be 3D after squeeze")
        skeleton = skeletonize(coronary_mask.astype(bool), method="lee").astype(bool)
        
        if not skeleton.any():
            raise ValueError("Unable to compute skeleton for coronary mask: empty skeleton")
        
        return skeleton

    def _find_skeleton_endpoints(self, skeleton_mask: np.ndarray) -> list[tuple[int, int, int]]:
        coords = np.argwhere(skeleton_mask)
        coord_set = {self._coord_tuple(coord) for coord in coords}
        endpoints: list[tuple[int, int, int]] = []
        for coord in coord_set:
            neighbor_count = 0
            for offset in self._neighbor_offsets():
                if (coord[0] + offset[0], coord[1] + offset[1], coord[2] + offset[2]) in coord_set:
                    neighbor_count += 1
            if neighbor_count <= 1:
                endpoints.append(coord)
        return endpoints

    def _compute_path_distance_map(self, skeleton_mask: np.ndarray, start_voxel: tuple[int, int, int]) -> torch.Tensor:
        coords = np.argwhere(skeleton_mask)
        if len(coords) == 0:
            raise ValueError("Empty skeleton")
        
        # map coord tuple -> index
        coord_to_index = {self._coord_tuple(coord): idx for idx, coord in enumerate(coords)}
        start_node = self._nearest_skeleton_voxel(start_voxel, skeleton_mask)
        start_idx = coord_to_index.get(start_node)
        if start_idx is None:
            # fallback: pick nearest skeleton coord
            start_idx = 0

        # Dijkstra on skeleton graph (26-neighborhood)
        n = len(coords)
        distances = np.full(n, np.inf, dtype=np.float32)
        distances[start_idx] = 0.0
        queue: list[tuple[float, int]] = [(0.0, start_idx)]
        while queue:
            dist, idx = heapq.heappop(queue)
            if dist > distances[idx]:
                continue
            x, y, z = self._coord_tuple(coords[idx])
            for offset in self._neighbor_offsets():
                neighbor = (x + offset[0], y + offset[1], z + offset[2])
                neighbor_idx = coord_to_index.get(neighbor)
                if neighbor_idx is None:
                    continue
                step = float(np.linalg.norm(np.asarray(offset, dtype=np.float32)))
                new_dist = dist + step / max(self.velocity, 1e-6)
                if new_dist < distances[neighbor_idx]:
                    distances[neighbor_idx] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor_idx))

        skel_dist = np.zeros(skeleton_mask.shape, dtype=np.float32)
        for idx, coord in enumerate(coords):
            skel_dist[self._coord_tuple(coord)] = distances[idx]

        # Project skeleton distances back to full volume via nearest-skeleton assignment
        distance_result = distance_transform_edt(~skeleton_mask, return_distances=True, return_indices=True)
        nearest_indices = distance_result[1]  # type: ignore
        projected = skel_dist[nearest_indices[0], nearest_indices[1], nearest_indices[2]]

        # cache and return
        self._cached_distance_map = torch.from_numpy(projected).to(dtype=torch.float32)
        return self._cached_distance_map

    def _density_from_distance(self, distance_map: torch.Tensor, time: float) -> torch.Tensor:
        tau = distance_map
        x_in = self.alpha * (time - tau - self.t_in)
        x_out = self.alpha * (time - tau - self.t_out)
        pulse = torch.sigmoid(x_in) - torch.sigmoid(x_out)
        return (self.mu_water_dsa + (self.mu_idodine - self.mu_water_dsa) * pulse).to(torch.float32)

    def _nearest_skeleton_voxel(self, start_voxel: tuple[int, int, int], skeleton_mask: np.ndarray) -> tuple[int, int, int]:
        if skeleton_mask[start_voxel].all():
            return start_voxel
        coords = np.argwhere(skeleton_mask)
        if len(coords) == 0:
            return start_voxel
        distances = np.linalg.norm(coords.astype(np.float32) - np.asarray(start_voxel, dtype=np.float32), axis=1)
        nearest_coord = coords[int(np.argmin(distances))]
        return self._coord_tuple(nearest_coord)

    def _nearest_skeleton_voxel_to_region(self, region_mask: np.ndarray, skeleton_mask: np.ndarray) -> tuple[int, int, int]:
        coords = np.argwhere(skeleton_mask)
        if len(coords) == 0:
            raise ValueError("Unable to find a skeleton voxel")
        region_coords = np.argwhere(region_mask)
        if len(region_coords) == 0:
            nearest = coords[0]
        else:
            distances = np.linalg.norm(coords.astype(np.float32)[None, :, :] - region_coords.astype(np.float32)[:, None, :], axis=-1)
            best_region_idx, best_skel_idx = np.unravel_index(int(np.argmin(distances)), distances.shape)
            nearest = coords[best_skel_idx]
        return self._coord_tuple(nearest)

    @staticmethod
    def _coord_tuple(coord: np.ndarray) -> tuple[int, int, int]:
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
