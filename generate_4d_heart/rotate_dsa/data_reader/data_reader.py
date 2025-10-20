from typing import Protocol, TypeVar
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from torch.nn import functional as F
import cupy as cp
from cupyx.scipy.ndimage import label, center_of_mass
from torch.utils.dlpack import from_dlpack, to_dlpack
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image

from ...roi import ROI
from ..cardiac_phase import CardiacPhase


def _pre_load(file: Path) -> tuple[Nifti1Image, np.ndarray]:
    if not file.exists():
        raise FileNotFoundError(f"nii file not found: {file}")
    image_nii = nib_load(file)
    assert isinstance(image_nii, Nifti1Image)
    affine = image_nii.affine if image_nii.affine is not None else np.eye(4)
    return image_nii, affine

def load_nifti(file: Path, is_label: bool = False) -> tuple[torch.Tensor, np.ndarray]:
    """load nifti file as torch tensor (shape: 1, 1, D, H, W) and return its affine matrix"""
    img, affine = _pre_load(file)
    tensor = torch.from_numpy(img.get_fdata())
    if is_label:
        tensor = tensor.round().to(torch.uint8)
    else:
        tensor = tensor.to(torch.float32)
    tensor = tensor[None][None]  # add batch and channel dim
    return tensor, affine

def load_nifti_with_roi_crop(file: Path, roi: ROI, is_label: bool = False) -> tuple[torch.Tensor, np.ndarray]:
    img, affine = _pre_load(file)
    img = roi.crop(img)
    tensor = torch.from_numpy(img.get_fdata())
    if is_label:
        tensor = tensor.round().to(torch.int8)
    else:
        tensor = tensor.to(torch.float32)
    tensor = tensor[None][None]  # add batch and channel dim
    affine = img.affine if img.affine is not None else np.eye(4)
    return tensor, affine

def load_and_zoom_dvf(
    dvf_path: Path, 
    roi: ROI, 
    operating_device: torch.device,
    final_device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    read dvf from nifti file and zoom it to the roi size
    """
    dvf_nii, _ = _pre_load(dvf_path)
    dvf_tensor = torch.from_numpy(dvf_nii.get_fdata())
    dvf_tensor = dvf_tensor.to(operating_device).to(torch.float)  # (H,W,D,3)
    dvf_tensor = dvf_tensor.squeeze().permute(3, 0, 1, 2)[None] # (1,3,H,W,D)
    dvf_tensor = F.interpolate(dvf_tensor, size=roi.get_crop_size(), mode='trilinear', align_corners=False)
    
    # image saved as spacing of 1mm, so we need to zoom it back to the original spacing
    zoom_rate = (1 / roi.get_zoom_rate()).flatten().tolist()
    for i in range(3):
        dvf_tensor[:, i] = dvf_tensor[:, i] * zoom_rate[i]
    return dvf_tensor.to(final_device)


def separate_coronary(coronary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    separate coronary to LCA(1) and RCA(2)
    Args:
        coronary (torch.Tensor): coronary segmentation
    Returns:
        lca (torch.Tensor): LCA segmentation, dtype: torch.bool
        rca (torch.Tensor): RCA segmentation, dtype: torch.bool
    """
    shape = coronary.shape
    coronary = coronary.squeeze()
    assert coronary.ndim == 3, "Coronary tensor after squeeze must be in shape (D, H, W)"
    
    coronary = cp.from_dlpack(to_dlpack(coronary.cuda()))
    labeled_array, num_features = label(coronary.astype(cp.int8))  # type: ignore
    
    if num_features <= 1:
        raise ValueError("Coronary segmentation must have at least 2 components")
    
    assert labeled_array is not None
    component_sizes = cp.bincount(labeled_array.ravel())[1:]  # Skip background (0)
    
    largest_indices = cp.argsort(component_sizes)[-2:][::-1] + 1
    
    region_0 = (labeled_array == largest_indices[0]).astype(cp.bool_)
    region_1 = (labeled_array == largest_indices[1]).astype(cp.bool_)
    
    center_0 = center_of_mass(region_0)
    center_1 = center_of_mass(region_1)
    
    region_0 = from_dlpack(region_0.toDlpack()).reshape(shape).to(torch.bool)
    region_1 = from_dlpack(region_1.toDlpack()).reshape(shape).to(torch.bool)
    
    if center_0[0] > center_1[0]:
        return region_0, region_1
    else:
        return region_1, region_0


@dataclass
class DataReaderResult:
    """ Holds the result of reading data for a specific phase. Labels are int8 tensors. tensors are in shape (B=1, C=1, D, H, W)."""
    phase: CardiacPhase
    volume: torch.Tensor
    cavity_label: torch.Tensor
    lca_label: torch.Tensor
    rca_label: torch.Tensor
    affine: np.ndarray
    
    def __post_init__(self):
        assert self.volume.ndim == 5 and self.volume.shape[0] == 1 and self.volume.shape[1] == 1, "Volume tensor must be in shape (1, 1, D, H, W)"
        assert self.cavity_label.shape == self.volume.shape, "Cavity label shape must match volume shape"
        assert self.lca_label.shape == self.volume.shape, "LCA label shape must match volume shape"
        assert self.rca_label.shape == self.volume.shape, "RCA label shape must match volume shape"
        assert self.cavity_label.dtype == torch.uint8, "Cavity label must be of type int8"
        assert self.lca_label.dtype == torch.bool, "LCA label must be of type bool"
        assert self.rca_label.dtype == torch.bool, "RCA label must be of type bool"
    
    def get_lca_center(self) -> tuple[float, float, float]:
        """Get the center of mass of the LCA label in (z, y, x) format"""
        coords = torch.nonzero(self.lca_label[0, 0], as_tuple=False)
        if coords.numel() == 0:
            return (-1.0, -1.0, -1.0)
        center = coords.float().mean(dim=0)
        return (center[0].item(), center[1].item(), center[2].item())
    
    def get_rca_center(self) -> tuple[float, float, float]:
        """Get the center of mass of the RCA label in (z, y, x) format"""
        coords = torch.nonzero(self.rca_label[0, 0], as_tuple=False)
        if coords.numel() == 0:
            return (-1.0, -1.0, -1.0)
        center = coords.float().mean(dim=0)
        return (center[0].item(), center[1].item(), center[2].item())
    
    
    def to_device(self, device: torch.device) -> "DataReaderResult":
        """Move all tensors to the specified device"""
        return DataReaderResult(
            phase=self.phase,
            volume=self.volume.to(device),
            cavity_label=self.cavity_label.to(device),
            lca_label=self.lca_label.to(device),
            rca_label=self.rca_label.to(device),
            affine=self.affine
        )
    
    def float(self) -> "DataReaderResult":
        """Convert all tensors in result to float32"""
        return DataReaderResult(
            phase=self.phase,
            volume=self.volume.to(torch.float32),
            cavity_label=self.cavity_label.to(torch.float32),
            lca_label=self.lca_label.to(torch.float32),
            rca_label=self.rca_label.to(torch.float32),
            affine=self.affine
        )
    
    def save(self, output_dir: Path):
        """Save all tensors to the specified directory"""
        from ...saver import save_nii
        output_dir.mkdir(exist_ok=True, parents=True)
        
        save_nii(output_dir / f"{self.phase}" / "volume.nii.gz", self.volume, self.affine)
        save_nii(output_dir / f"{self.phase}" / "cavity_label.nii.gz", self.cavity_label, self.affine, is_label=True)
        save_nii(output_dir / f"{self.phase}" / "lca_label.nii.gz", self.lca_label, self.affine, is_label=True)
        save_nii(output_dir / f"{self.phase}" / "rca_label.nii.gz", self.rca_label, self.affine, is_label=True)

# TODO 用 Dataset/Dataloader 的形式实现 reader
class DataReader(Protocol):
    n_phases: int
    origin_image_size: tuple[int, int, int]
    origin_image_affine: np.ndarray

    def get_data(self, phase: CardiacPhase) -> DataReaderResult:
        ...