from typing import Protocol

import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt, binary_dilation, gaussian_filter
import torch
from torch.utils.dlpack import to_dlpack as tensor2dlpack
from torch.utils.dlpack import from_dlpack as dlpack2tensor

from .. import LV_LABEL, LV_MYO_LABEL

class MovementEnhancer(Protocol):
    def __init__(self, cavity_label: torch.Tensor, coronary_label: torch.Tensor) -> None:
        ...

    def __call__(self, dvf: torch.Tensor) -> torch.Tensor:
        ...


class CoronaryBoundLV(MovementEnhancer):
    def __init__(
        self, 
        cavity_label: torch.Tensor, 
        coronary_label: torch.Tensor,
        enhance_weight_at_myo_external_contour: float = 0.5
    ) -> None:
        lv = (cavity_label.to(torch.uint8) == LV_LABEL).squeeze()
        myo = (cavity_label.to(torch.uint8) == LV_MYO_LABEL).squeeze()
        
        lv_cp = cp.from_dlpack(tensor2dlpack(lv.cuda())).astype(cp.bool_)
        myo_cp = cp.from_dlpack(tensor2dlpack(myo.cuda())).astype(cp.bool_)
        
        coronary_cp = cp.from_dlpack(tensor2dlpack(coronary_label.squeeze().cuda())).astype(cp.uint8)
        self.dilate_coronary_mask = binary_dilation(coronary_cp, structure=cp.ones((9,9,9))).astype(cp.bool_)
        self.smooth_mask = binary_dilation(self.dilate_coronary_mask).astype(cp.bool_)
        
        dist: cp.ndarray
        indices: cp.ndarray
        dist, indices = distance_transform_edt(~lv_cp, return_distances=True, return_indices=True) # type: ignore
        self.indices = indices[:, self.dilate_coronary_mask]  # shape = (3, N), N is the total numpy of myo voxels

        dist_myo_max = dist[myo_cp].max()
        alpha = -dist_myo_max / cp.log(enhance_weight_at_myo_external_contour)
        self.alpha = cp.exp( - dist / alpha )
        self.alpha = self.alpha[self.dilate_coronary_mask]

    
    def __call__(self, dvf: torch.Tensor) -> torch.Tensor:
        dvf_cp = cp.from_dlpack(tensor2dlpack(dvf.cuda()))
        dvf_cp[:,:, self.dilate_coronary_mask] = (
            dvf_cp[:, :, *self.indices] * self.alpha
            + dvf_cp[:, :, self.dilate_coronary_mask] * (1 - self.alpha)
        )
        smoothed = cp.zeros_like(dvf_cp)
        for i in range(3):
            smoothed[:, i] = gaussian_filter(dvf_cp[:, i], sigma=2.0)
        dvf_cp[:, :, self.smooth_mask] = smoothed[:, :, self.smooth_mask]
        
        dvf = dlpack2tensor(dvf_cp.toDlpack()).cpu()
        return dvf