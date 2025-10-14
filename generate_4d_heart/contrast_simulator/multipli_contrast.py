import torch
import numpy as np
from scipy.ndimage import binary_dilation

from .contrast_simulator import ContrastSimulator


class MultipliContrast(ContrastSimulator):
    def __init__(
        self, 
        coronary_factor: float = 5.0,
        cavity_factor: float = 0.7
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
        if res.min() < 0:
            res -= res.min()
            # TODO 这里在处理 ASOCA 图像时会直接 + 3072 导致后续衰减和冠脉增强的幅度都变大了
            # 虽然效果确实还不错，但需要考虑是否是合理的

        heart_label = binary_dilation((coronary_label > 0).cpu().numpy().astype(np.uint8))
        heart_label = torch.from_numpy(heart_label).to(coronary_label.device)
        assert cavity_label.dtype == torch.uint8
        assert coronary_label.dtype == torch.bool
        res[heart_label>0 & (~coronary_label)] *= self.cavity_factor
        res[coronary_label] *= self.coronary_factor
        return res