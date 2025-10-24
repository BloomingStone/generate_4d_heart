from typing import Protocol

import torch

class ContrastSimulator(Protocol):
    """
    Simulate the affect of DSA contrast of coronary artery (LCA/RCA) based on given image and label
    """
    def simulate(
        self, 
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
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
        