from typing import Protocol

import torch

class MovementEnhancer(Protocol):
    cavity_label: torch.Tensor
    coronary_label: torch.Tensor
    
    def __init__(self, cavity_label: torch.Tensor, coronary_label: torch.Tensor) -> None:
        self.cavity_label = cavity_label
        self.coronary_label = coronary_label

    def enhance(self, dvf: torch.Tensor) -> torch.Tensor:
        ...

class IdentityMovementEnhancer(MovementEnhancer):
    def __init__(self, cavity_label: torch.Tensor, coronary_label: torch.Tensor) -> None:
        self.cavity_label = cavity_label
        self.coronary_label = coronary_label
    
    def enhance(self, dvf: torch.Tensor) -> torch.Tensor:
        return dvf