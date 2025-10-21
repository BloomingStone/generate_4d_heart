import torch
import numpy as np
from scipy.ndimage import binary_dilation

from .contrast_simulator import ContrastSimulator


class MultipliContrast(ContrastSimulator):
    def __init__(
        self, 
        coronary_factor: float = 15.0,
        cavity_factor: float = 0.3
    ):
        """
        adjust the contrast of coronary and cavity by simple multiplication
        """
        self.coronary_factor = coronary_factor
        self.cavity_factor = cavity_factor
    
    def simulate(
        self, 
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
    ) -> torch.Tensor:
        res = ori_volume.clone()
        sentinel_mask = (res <= -2000)  # Some CTs may use -3023 or -2000 as 'sentinel' to mark invalid voxels
        min_value = res[~sentinel_mask].min()
        res[sentinel_mask] = min_value
        if res.min() < 0:
            res -= res.min()

        heart_label = binary_dilation((coronary_label > 0).cpu().numpy().astype(np.uint8))
        heart_label = torch.from_numpy(heart_label).to(coronary_label.device)
        assert cavity_label.dtype == torch.uint8
        assert coronary_label.dtype == torch.bool
        res[heart_label>0 & (~coronary_label)] *= self.cavity_factor
        res[coronary_label] *= self.coronary_factor
        return res