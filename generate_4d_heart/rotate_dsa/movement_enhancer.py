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
        dilate_coronary_mask = binary_dilation(coronary_cp, structure=cp.ones((9,9,9))).astype(cp.bool_)
        smooth_mask = binary_dilation(dilate_coronary_mask).astype(cp.bool_)
        
        dist: cp.ndarray
        indices: cp.ndarray
        dist, indices = distance_transform_edt(~lv_cp, return_distances=True, return_indices=True) # type: ignore
        self.indices = indices[:, dilate_coronary_mask]  # shape = (3, N), N is the total numpy of coronary voxels

        dist_myo_max = dist[myo_cp].max()
        alpha = -dist_myo_max / cp.log(enhance_weight_at_myo_external_contour)
        self.alpha = cp.exp( - dist / alpha )
        self.alpha = self.alpha[dilate_coronary_mask]
        
        self.dilate_coronary_indices = cp.where(dilate_coronary_mask)
        self.smooth_indices = cp.where(smooth_mask)

    def __call__(self, dvf: torch.Tensor) -> torch.Tensor:
        dvf_cp = cp.from_dlpack(tensor2dlpack(dvf.cuda()))
        dvf_cp[:,:, *self.dilate_coronary_indices] = (
            dvf_cp[:, :, *self.indices] * self.alpha
            + dvf_cp[:, :, *self.dilate_coronary_indices] * (1 - self.alpha)
        )
        smoothed = cp.zeros_like(dvf_cp)
        for i in range(3):
            smoothed[:, i] = gaussian_filter(dvf_cp[:, i], sigma=2.0)
        dvf_cp[:, :, *self.smooth_indices] = smoothed[:, :, *self.smooth_indices]
        
        dvf = dlpack2tensor(dvf_cp.toDlpack()).cpu()
        del dvf_cp
        return dvf

class CoronaryBoundLVLinear(MovementEnhancer):
    def __init__(
        self, 
        cavity_label: torch.Tensor, 
        coronary_label: torch.Tensor,
        enhance_weight_at_farthest_coroanry: float = 0.1
    ) -> None:
        lv = (cavity_label.to(torch.uint8) == LV_LABEL).squeeze()
        lv_cp = cp.from_dlpack(tensor2dlpack(lv.cuda())).astype(cp.bool_)
        
        coronary_cp = cp.from_dlpack(tensor2dlpack(coronary_label.squeeze().cuda())).astype(cp.uint8)
        dilate_coronary_mask = binary_dilation(coronary_cp, structure=cp.ones((9,9,9))).astype(cp.bool_)
        smooth_mask = binary_dilation(dilate_coronary_mask).astype(cp.bool_)
        
        dist: cp.ndarray
        indices: cp.ndarray
        dist, indices = distance_transform_edt(~lv_cp, return_distances=True, return_indices=True) # type: ignore
        self.indices = indices[:, dilate_coronary_mask]  # shape = (3, N), N is the total numpy of coronary voxels

        dist_coronary_max = dist[dilate_coronary_mask].max()
        alpha = (1 - enhance_weight_at_farthest_coroanry) / dist_coronary_max
        self.alpha = cp.clip(1 - dist * alpha, 0.0, 1.0)
        self.alpha = self.alpha[dilate_coronary_mask]
        
        self.dilate_coronary_indices = cp.where(dilate_coronary_mask)
        self.smooth_indices = cp.where(smooth_mask)

    def __call__(self, dvf: torch.Tensor) -> torch.Tensor:
        dvf_cp = cp.from_dlpack(tensor2dlpack(dvf.cuda()))
        dvf_cp[:,:, *self.dilate_coronary_indices] = (
            dvf_cp[:, :, *self.indices] * self.alpha
            + dvf_cp[:, :, *self.dilate_coronary_indices] * (1 - self.alpha)
        )
        smoothed = cp.zeros_like(dvf_cp)
        for i in range(3):
            smoothed[:, i] = gaussian_filter(dvf_cp[:, i], sigma=2.0)
        dvf_cp[:, :, *self.smooth_indices] = smoothed[:, :, *self.smooth_indices]
        
        dvf = dlpack2tensor(dvf_cp.toDlpack()).cpu()
        del dvf_cp
        return dvf