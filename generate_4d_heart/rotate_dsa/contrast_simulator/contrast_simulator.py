from typing import Protocol

import numpy as np
import torch
from generate_4d_heart import CavityLabel, MU_WATER, MU_IDODINE

class ContrastSimulator(Protocol):
        
    contrast_change_over_time: bool = False
    
    """
    Simulate the affect of DSA contrast of coronary artery (LCA/RCA) based on given image and label
    """
    def preprocess(
        self, 
        ori_volume: torch.Tensor, 
        cavity_label: torch.Tensor,
        affine: np.ndarray,
    ) -> torch.Tensor:
        """
        Preprocess the input volume, e.g. remove contrast in CTA volume, reset invalid values, map raw HU values to attenuation coefficients, etc. 
        This will be called before `simulate` or `simulate_with_time`.
        """
        ...
    
    def simulate(
        self, 
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        affine: np.ndarray
    ) -> torch.Tensor:
        """Simulate the contrast effect on the original volume. This will be called after `preprocess` and should return the simulated volume."""
        ...
    
    def simulate_with_time(
        self, 
        time: float,
        ori_volume: torch.Tensor, 
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        affine: np.ndarray
    ) -> torch.Tensor:
        """
        Simulate the contrast effect on the original volume with time information. This will be called after `preprocess` and should return the simulated volume. 
        Usually, this should only be implemented when `contrast_change_over_time` is True.
        """
        ...

class IdentityContrast(ContrastSimulator):
    """A simple contrast simulator that does not change the input volume, used for ablation study to verify the effect of contrast simulation."""
    def preprocess(
        self, 
        ori_volume: torch.Tensor, 
        cavity_label: torch.Tensor,
        affine: np.ndarray,
    ) -> torch.Tensor:
        return ori_volume
    
    def simulate(
        self, 
        ori_volume: torch.Tensor, 
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        affine: np.ndarray,
    ) -> torch.Tensor:
        return ori_volume

    def simulate_with_time(
        self,
        time: float,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        affine: np.ndarray
    ) -> torch.Tensor:
        # For identity, ignore time and return original volume
        import warnings
        warnings.warn("IdentityContrast does not support contrast change over time, `simulate_with_time` will ignore `time` input and return the same result as `simulate`")
        return self.simulate(ori_volume, cavity_label, coronary_label, affine)


class SimplePreprocessContrast(ContrastSimulator):
    """
    A simple contrast simulator that only preprocesses the input volume by mapping HU values to attenuation coefficients, 
    without adding any contrast effect. 
    This can be used to verify the effect of preprocessing alone.
    """
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
        # StaticIodineContrast is static by design
        assert self.contrast_change_over_time == False, "SimplePreprocessContrast does not support contrast change over time"
        # `ori_volume` is expected to be a preprocessed attenuation-like baseline
        return ori_volume


    
    def simulate_with_time(
        self, 
        time: float,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        affine: np.ndarray
    ) -> torch.Tensor:
        import warnings
        warnings.warn("SimplePreprocessContrast does not support contrast change over time, `simulate_with_time` will ignore `time` input and return the same result as `simulate`")
        return self.simulate(ori_volume, cavity_label, coronary_label, affine)
