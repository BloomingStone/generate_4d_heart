import torch

from .contrast_simulator import ContrastSimulator
from ... import LA_LABEL, MU_WATER, MU_IDODINE

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
    
    def simulate(
        self, 
        ori_volume: torch.Tensor,   # in HU
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
    ) -> torch.Tensor:
        res = ori_volume.clone()
        res = res / 1000.0 * self.mu_water_dsa + self.mu_water_dsa  # 将 HU 转换为衰减系数
        
        assert cavity_label.dtype == torch.uint8
        assert coronary_label.dtype == torch.bool
        
        # HU 值在 v_min-v_max 之间的部分, 为心腔及被对比剂增强过的, 现在恢复为水衰减系数
        masked_volume = ori_volume[cavity_label == LA_LABEL]  # 使用左房，因此此区域肌肉结构占比少，大部分是造影血流
        v_min = torch.quantile(masked_volume, 0.1 / 100)
        v_max = torch.quantile(masked_volume, 99.9 / 100)
        threshold_mask = (ori_volume > v_min) & (ori_volume < v_max)
        
        res[
            (cavity_label>0) 
            | threshold_mask
        ] = self.mu_water_dsa
        
        # 对冠状动脉部分, 设为碘化钠对比剂的衰减系数
        res[coronary_label] = self.mu_idodine
        
        # 一些图像中会将 无效值标记为 -3096, 将这部分的衰减系数设置为 0
        res[ori_volume < -2000] = 0
        
        return res
