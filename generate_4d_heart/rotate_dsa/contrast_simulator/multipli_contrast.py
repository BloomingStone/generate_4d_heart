import torch
import numpy as np

from .contrast_simulator import ContrastSimulator
from generate_4d_heart import CavityLabel, MU_WATER, MU_IDODINE

class MultipliContrast(ContrastSimulator):
    def __init__(
        self, 
        mu_water_dsa: float = MU_WATER,    # 水衰减系数 (mm^-1)
        mu_idodine: float = MU_IDODINE,    # 碘化钠对比剂衰减系数 (mm^-1)
    ):
        """
        adjust the contrast of coronary and cavity by simple multiplication
        """
        self.mu_idodine = mu_idodine
        self.mu_water_dsa = mu_water_dsa

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
        res = res / 1000.0 * self.mu_water_dsa + self.mu_water_dsa  # HU -> attenuation

        assert cavity_label.dtype == torch.uint8

        # Use left atrium region to estimate typical enhanced range
        masked_volume = ori_volume.squeeze()[cavity_label.squeeze() == CavityLabel.LA]
        if masked_volume.numel() > 0:
            v_min = torch.quantile(masked_volume, 0.1 / 100)
            v_max = torch.quantile(masked_volume, 99.9 / 100)
            threshold_mask = (ori_volume > v_min) & (ori_volume < v_max)
            res[(cavity_label > 0) | threshold_mask] = self.mu_water_dsa

        # Invalid HU values -> zero attenuation
        res[ori_volume < -2000] = 0.0
        return res
    
    def simulate(
        self, 
        ori_volume: torch.Tensor,   # assumed preprocessed baseline (attenuation-like)
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        affine: np.ndarray,
    ) -> torch.Tensor:
        # MultipliContrast is static by design
        assert self.contrast_change_over_time == False, "MultipliContrast does not support contrast change over time"
        # `ori_volume` is expected to be a preprocessed attenuation-like baseline
        res = ori_volume.clone()
        assert cavity_label.dtype == torch.uint8
        assert coronary_label.dtype == torch.bool or coronary_label.dtype == torch.uint8

        # Coronary voxels should be set to iodine attenuation
        res[coronary_label > 0] = self.mu_idodine
        return res


    
    def simulate_with_time(
        self, 
        time: float,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        affine: np.ndarray
    ) -> torch.Tensor:
        import warnings
        warnings.warn("MultipliContrast does not support contrast change over time, `simulate_with_time` will ignore `time` input and return the same result as `simulate`")
        return self.simulate(ori_volume, cavity_label, coronary_label, affine)
