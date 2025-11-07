import torch

from .contrast_simulator import ContrastSimulator

class MultipliContrast(ContrastSimulator):
    def __init__(
        self, 
        coronary_factor: float = 7.0,
        cavity_factor: float = 0.7,
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
        res = ori_volume
        assert cavity_label.dtype == torch.uint8
        assert coronary_label.dtype == torch.bool
        coronary_mean = res[coronary_label].mean()
        
        bone_HU_min = 500 + 1024
        
        if ori_volume.max() > bone_HU_min:
            bone_area = ori_volume > bone_HU_min
            bone_max = res[bone_area].max()
            bone_factor = coronary_mean / bone_max * self.coronary_factor * 0.5
            res[bone_area] *= bone_factor
        
        res[cavity_label>0 & (~coronary_label)] *= self.cavity_factor
        res[coronary_label] = coronary_mean * self.coronary_factor
        
        
        return res