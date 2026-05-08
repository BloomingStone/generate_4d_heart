from typing import Protocol

import torch

class ContrastSimulator(Protocol):
        
    contrast_change_over_time: bool = False
    
    """
    Simulate the affect of DSA contrast of coronary artery (LCA/RCA) based on given image and label
    """
    def preprocess(
        self, 
        ori_volume: torch.Tensor, 
        cavity_label: torch.Tensor,
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
        coronary_label: torch.Tensor
    ) -> torch.Tensor:
        """Simulate the contrast effect on the original volume. This will be called after `preprocess` and should return the simulated volume."""
        ...
    
    def simulate_with_time(
        self, 
        ori_volume: torch.Tensor, 
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        time: float
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
    ) -> torch.Tensor:
        return ori_volume
    
    def simulate(
        self, 
        ori_volume: torch.Tensor, 
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor
    ) -> torch.Tensor:
        return ori_volume

    def simulate_with_time(
        self,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        time: float
    ) -> torch.Tensor:
        # For identity, ignore time and return original volume
        import warnings
        warnings.warn("IdentityContrast does not support contrast change over time, `simulate_with_time` will ignore `time` input and return the same result as `simulate`")
        return self.simulate(ori_volume, cavity_label, coronary_label)
        