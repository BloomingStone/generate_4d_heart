import torch
from torch import Tensor
from torch.nn import functional as F

from .contrast_simulator import ContrastSimulator



def median_filter3d(volume: Tensor, kernel_size: int = 3) -> Tensor:
    """
    3D median filter for a 5D tensor: [B, C, D, H, W]
    """
    assert volume.dim() == 5, "Input must be [B, C, D, H, W]"
    pad = kernel_size // 2

    vol = F.pad(volume, (pad, pad, pad, pad, pad, pad), mode="replicate")
    
    # unfold in depth (D), height (H), width (W)
    patches = vol.unfold(2, kernel_size, 1) \
                 .unfold(3, kernel_size, 1) \
                 .unfold(4, kernel_size, 1)  # → [B,C,D,H,W,k,k,k]

    # reshape patches so last dimension is kernel volume
    patches = patches.reshape(volume.shape + (-1,))  # kernel_size^3

    # median in the last dim
    filtered = patches.median(dim=-1).values

    return filtered


class MultipliContrast(ContrastSimulator):
    def __init__(
        self, 
        coronary_factor: float = 5.0,   # DSA 所用X射线能量更小，且与碘的 K-edge 相近，因此吸收率会显著增加
        cavity_factor: float = 0.3,     # CTA下心腔和主动脉中有造影剂充盈，需要降低对比度
        mu_water: float = 0.020,    # 水衰减系数 (mm^-1)
    ):
        """
        adjust the contrast of coronary and cavity by simple multiplication
        """
        self.coronary_factor = coronary_factor
        self.cavity_factor = cavity_factor
        self.mu_water = mu_water
    
    def simulate(
        self, 
        ori_volume: torch.Tensor,   # in HU
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
    ) -> torch.Tensor:
        res = ori_volume.clone()
        res = res / 1000.0 * self.mu_water + self.mu_water  # 将 HU 转换为衰减系数
        
        assert cavity_label.dtype == torch.uint8
        assert coronary_label.dtype == torch.bool
        
        # HU 值在 v_min-v_max 之间的部分, 为心腔及被对比剂增强过的，需要乘以 cavity_factor
        v_min = torch.quantile(ori_volume[cavity_label], 0.1 / 100)
        v_max = torch.quantile(ori_volume[cavity_label], 99.9 / 100)
        res[
            (cavity_label>0) 
            | ((ori_volume > v_min) & (ori_volume < v_max))
        ] *= self.cavity_factor
        
        res[coronary_label] *= self.coronary_factor / self.cavity_factor
        
        # 一些图像中会将 无效值标记为 -3096, 将这部分的衰减系数设置为 0
        res[ori_volume < -2000] = 0
        
        return res
