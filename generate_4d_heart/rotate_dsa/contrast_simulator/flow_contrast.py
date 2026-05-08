from __future__ import annotations

from dataclasses import dataclass, field
import heapq

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from generate_4d_heart import CavityLabel, MU_IDODINE, MU_WATER, VesselLabel

from .contrast_simulator import ContrastSimulator

from skimage.morphology import skeletonize


@dataclass
class FlowContrast(ContrastSimulator):
    """Time-varying coronary contrast simulator.

    The start voxel is cached once because it is assumed to be anatomically stable.
    The coronary skeleton is recomputed on every call because the vessel can deform.
    The output density baseline follows the same HU-to-attenuation style as MultipliContrast.
    """

    contrast_change_over_time: bool = True

    # Baseline density parameters, mirroring MultipliContrast's attenuation scale.
    mu_water_dsa: float = MU_WATER
    mu_idodine: float = MU_IDODINE

    # Time model parameters for rho(d, t)
    velocity: float = 200
    t_in: float = 0.15
    t_out: float = 2.5
    alpha: float = 5.0

    # optional manual entry seed in voxel coordinates (x, y, z)
    entry_seed: tuple[int, int, int] | None = None
    _cached_start_voxel: tuple[int, int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._cached_start_voxel = self.entry_seed

    def simulate(
        self,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("FlowContrast does not support simulate without time, please use `simulate_with_time` instead")

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

        cavity = cavity_label.squeeze().cpu().numpy()
        coronary_mask = coronary_label.squeeze().to(torch.bool).cpu().numpy()
        if not coronary_mask.any():
            return self._base_density_map(ori_volume, cavity_label, coronary_label)

        start_voxel = self._resolve_start_voxel(cavity, coronary_mask)
        skeleton_mask = self._compute_skeleton(coronary_mask)
        if not skeleton_mask.any():
            return self._base_density_map(ori_volume, cavity_label, coronary_label)

        distance_map = self._compute_path_distance_map(skeleton_mask, start_voxel)
        coronary_density = self._density_from_distance(distance_map, float(time))

        density = self._base_density_map(ori_volume, cavity_label, coronary_label)
        density_np = density.squeeze(0).squeeze(0).cpu().numpy()
        density_np[coronary_mask] = coronary_density[coronary_mask]
        return torch.from_numpy(density_np).to(device=ori_volume.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def _base_density_map(
        self,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
    ) -> torch.Tensor:
        """Convert HU to attenuation-like density using the same baseline logic as MultipliContrast."""
        density = ori_volume.clone().to(torch.float32)
        density = density / 1000.0 * self.mu_water_dsa + self.mu_water_dsa

        cavity = cavity_label.squeeze().to(torch.uint8)
        coronary = coronary_label.squeeze().to(torch.bool)

        masked_volume = ori_volume.squeeze()[cavity == CavityLabel.LA]
        if masked_volume.numel() > 0:
            v_min = torch.quantile(masked_volume, 0.1 / 100)
            v_max = torch.quantile(masked_volume, 99.9 / 100)
            threshold_mask = (ori_volume.squeeze() > v_min) & (ori_volume.squeeze() < v_max)
            density.squeeze()[cavity > 0] = self.mu_water_dsa
            density.squeeze()[threshold_mask] = self.mu_water_dsa

        density.squeeze()[coronary] = self.mu_water_dsa
        density.squeeze()[ori_volume.squeeze() < -2000] = 0.0
        return density

    def _resolve_start_voxel(self, cavity: np.ndarray, coronary_mask: np.ndarray) -> tuple[int, int, int]:
        if self._cached_start_voxel is not None:
            return self._cached_start_voxel

        skeleton_mask = self._compute_skeleton(coronary_mask)
        if not skeleton_mask.any():
            raise ValueError("Unable to resolve a start voxel for FlowContrast: empty skeleton")

        aorta_mask = cavity == int(VesselLabel.AORTA)
        
        if not aorta_mask.any():
            raise ValueError("Unable to resolve a start voxel for FlowContrast: empty aorta region")
        
        start = self._nearest_skeleton_voxel_to_region(aorta_mask, skeleton_mask)

        self._cached_start_voxel = start
        return start

    def _compute_skeleton(self, coronary_mask: np.ndarray) -> np.ndarray:
        if coronary_mask.ndim != 3:
            raise ValueError("coronary_label must be 3D after squeeze")
        return skeletonize(coronary_mask.astype(bool), method="lee").astype(bool)

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

    def _compute_path_distance_map(self, skeleton_mask: np.ndarray, start_voxel: tuple[int, int, int]) -> np.ndarray:
        coords = np.argwhere(skeleton_mask)
        if len(coords) == 0:
            raise ValueError("Empty skeleton")
        distances = np.linalg.norm(coords - np.array(start_voxel, dtype=np.float32), axis=1).astype(np.float32) / max(self.velocity, 1e-6)
        distance_map = np.zeros(skeleton_mask.shape, dtype=np.float32)
        distance_map[tuple(coords.T)] = distances

        # coord_to_index = {self._coord_tuple(coord): idx for idx, coord in enumerate(coords)}
        # start_voxel = self._nearest_skeleton_voxel(start_voxel, skeleton_mask)
        # start_idx = coord_to_index[start_voxel]

        # distances = np.full(len(coords), np.inf, dtype=np.float32)
        # distances[start_idx] = 0.0
        # queue: list[tuple[float, int]] = [(0.0, start_idx)]

        # while queue:
        #     dist, idx = heapq.heappop(queue)
        #     if dist > distances[idx]:
        #         continue
        #     x, y, z = self._coord_tuple(coords[idx])
        #     for offset in self._neighbor_offsets():
        #         neighbor = (x + offset[0], y + offset[1], z + offset[2])
        #         neighbor_idx = coord_to_index.get(neighbor)
        #         if neighbor_idx is None:
        #             continue
        #         step = float(np.linalg.norm(np.asarray(offset, dtype=np.float32)))
        #         new_dist = dist + step / max(self.velocity, 1e-6)
        #         if new_dist < distances[neighbor_idx]:
        #             distances[neighbor_idx] = new_dist
        #             heapq.heappush(queue, (new_dist, neighbor_idx))

        # distance_map = np.zeros(skeleton_mask.shape, dtype=np.float32)
        # for idx, coord in enumerate(coords):
        #     distance_map[self._coord_tuple(coord)] = distances[idx]

        # Project the skeleton distances back to every voxel by nearest-skeleton assignment.
        distance_result = distance_transform_edt(~skeleton_mask, return_distances=True, return_indices=True)
        nearest_indices = distance_result[1]    # type: ignore
        projected = np.zeros_like(distance_map)
        projected[:] = distance_map[nearest_indices[0], nearest_indices[1], nearest_indices[2]]
        return projected

    def _density_from_distance(self, distance_map: np.ndarray, time: float) -> np.ndarray:
        tau = distance_map
        x_in = self.alpha * (time - tau - self.t_in)
        x_out = self.alpha * (time - tau - self.t_out)
        pulse = torch.sigmoid(torch.from_numpy(x_in)).numpy() - torch.sigmoid(torch.from_numpy(x_out)).numpy()
        return (self.mu_water_dsa + (self.mu_idodine - self.mu_water_dsa) * pulse).astype(np.float32)

    def _nearest_skeleton_voxel(self, start_voxel: tuple[int, int, int], skeleton_mask: np.ndarray) -> tuple[int, int, int]:
        if skeleton_mask[start_voxel]:
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
