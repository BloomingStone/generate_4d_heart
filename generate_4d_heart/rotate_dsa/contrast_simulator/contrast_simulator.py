from typing import Protocol

import torch

class ContrastSimulator(Protocol):
        
    contrast_change_over_time: bool = False
    
    """
    Simulate the affect of DSA contrast of coronary artery (LCA/RCA) based on given image and label
    """
    # def preprocess(
    #     self, 
    #     ori_volume: torch.Tensor, 
    #     cavity_label: torch.Tensor,
    # ) -> None:
    #     """
    #     Preprocess the input volume, e.g. remove contrast in CTA volume, reset invalid values, map raw HU values to attenuation coefficients, etc. 
    #     This will be called before `simulate` or `simulate_with_time`.
    #     """
    #     ...
    
    def simulate(
        self, 
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor
    ) -> torch.Tensor:
        ...
    
    def simulate_with_time(
        self, 
        ori_volume: torch.Tensor, 
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        time: float
    ) -> torch.Tensor:
        ...

class IdentityContrast(ContrastSimulator):
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
        