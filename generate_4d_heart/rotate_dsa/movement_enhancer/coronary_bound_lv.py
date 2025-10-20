from .movement_enhancer import MovementEnhancer
import torch

class CoronaryBoundLV(MovementEnhancer):
    def __init__(
        self, 
        cavity_label: torch.Tensor, 
        coronary_label: torch.Tensor
    ) -> None:
        ...