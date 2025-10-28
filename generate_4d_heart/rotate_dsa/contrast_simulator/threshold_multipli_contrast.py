import torch

from .contrast_simulator import ContrastSimulator


class ThresholdMultipliContrast(ContrastSimulator):
    def __init__(
        self, 
        lung_threshold: int = -600 + 1024,
        heart_threshold: int = 0 + 1024,
        bone_threshold: int = 600 + 1024,
        lung_alpha: float = 1,
        heart_alpha: float = 0.3,
        bone_alpha: float = 1.0,
        coronary_alpha: float = 12.0
    ):
        """
        adjust the contrast of coronary and cavity by simple multiplication
        """
        self.lung_threshold = lung_threshold
        self.heart_threshold = heart_threshold
        self.bone_threshold = bone_threshold
        self.lung_alpha = lung_alpha
        self.heart_alpha = heart_alpha
        self.bone_alpha = bone_alpha
        self.coronary_alpha = coronary_alpha
    
    def simulate(
        self, 
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
    ) -> torch.Tensor:
        density = ori_volume.clone()
        
        assert density.dtype == torch.float32
        assert coronary_label.dtype == torch.bool
        air = torch.where(density <= self.lung_threshold)
        lung = torch.where((self.lung_threshold < density) & (density <= self.heart_threshold))
        heart = torch.where((self.heart_threshold < density) & (density <= self.bone_threshold))
        bone = torch.where(density > self.bone_threshold)
        coronary = torch.where(coronary_label)

        coronary_value = density[coronary].clone()
        density[air] = density[lung].min() if len(lung[0]) > 0 else self.lung_threshold
        density[lung] *= self.lung_alpha
        density[heart] *= self.heart_alpha
        density[bone] *= self.bone_alpha
        density[coronary] = coronary_value * self.coronary_alpha
        
        return density