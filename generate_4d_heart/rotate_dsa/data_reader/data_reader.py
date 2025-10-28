from typing import Protocol, Literal
from dataclasses import dataclass, field
from pathlib import Path

import torch
import numpy as np
from torch.nn import functional as F
import cupy as cp
from cupyx.scipy.ndimage import label, center_of_mass
from skimage.morphology import skeletonize
from torch.utils.dlpack import from_dlpack as dlpack2tensor
from torch.utils.dlpack import to_dlpack as tensor2dlpack
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image

from ...roi import ROI
from ..cardiac_phase import CardiacPhase


def pre_load(file: Path) -> tuple[Nifti1Image, np.ndarray]:
    if not file.exists():
        raise FileNotFoundError(f"nii file not found: {file}")
    image_nii = nib_load(file)
    assert isinstance(image_nii, Nifti1Image)
    affine = image_nii.affine if image_nii.affine is not None else np.eye(4)
    return image_nii, affine

def load_nifti(file: Path, is_label: bool = False) -> tuple[torch.Tensor, np.ndarray]:
    """load nifti file as torch tensor (shape: 1, 1, D, H, W) and return its affine matrix"""
    img, affine = pre_load(file)
    tensor = torch.from_numpy(img.get_fdata())
    if is_label:
        tensor = tensor.round().to(torch.uint8)
    else:
        tensor = tensor.to(torch.float32)
    tensor = tensor[None][None]  # add batch and channel dim
    return tensor, affine

def load_nifti_with_roi_crop(file: Path, roi: ROI, is_label: bool = False) -> tuple[torch.Tensor, np.ndarray]:
    img, affine = pre_load(file)
    img = roi.crop(img)
    tensor = torch.from_numpy(img.get_fdata())
    if is_label:
        tensor = tensor.round().to(torch.uint8)
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
    dvf_nii, _ = pre_load(dvf_path)
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
    
    coronary = cp.from_dlpack(tensor2dlpack(coronary.cuda()))
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
    
    region_0 = dlpack2tensor(region_0.toDlpack()).reshape(shape).to(torch.bool)
    region_1 = dlpack2tensor(region_1.toDlpack()).reshape(shape).to(torch.bool)
    
    if center_0[0] > center_1[0]:
        return region_0, region_1
    else:
        return region_1, region_0

def recenter(
    original_affine: np.ndarray,
    center_voxel: tuple[int, int, int]
) -> np.ndarray:
    """get the affine that set the center of the image to the given center_voxel
    """
    B = original_affine[:3, :3]
    new_t = -(B @ np.array(center_voxel))
    T = np.eye(4)
    T[:3, :3] = B
    T[:3, 3] = new_t
    return T

def get_coronary_centering_affine(coronary_label: torch.Tensor, volume_affine: np.ndarray) -> np.ndarray:
    """
    Get the affine matrix of that may coroanry at ordinate of the world coordinate
    
    Args:
        coronary_label (torch.Tensor): coronary label (LCA or RCA) , shape: (1, 1, D, H, W)
        volume_affine (np.ndarray): affine matrix of the volume, shape: (4, 4)
    """
    
    label_center: tuple[int, int, int] = center_of_mass(cp.from_dlpack(tensor2dlpack(coronary_label.cuda()))) # type: ignore
    W, H, D = coronary_label.shape[-3:]
    image_center = (W/2, H/2, D/2)
    label_center_voxel = (
        int( (image_center[0] + label_center[0]) / 2 ), # left and right, set as the mean of image_center and label_center
        int( (image_center[1] + label_center[1]) / 2 ), # antero-posterior, same as above
        int( image_center[2] )                          # up and down, set as the center of image
    )
    
    return recenter(volume_affine, label_center_voxel)

@dataclass
class DataReaderResult:
    """ Holds the result of reading data for a specific phase. Labels are int8 tensors. tensors are in shape (B=1, C=1, D, H, W)."""
    phase: CardiacPhase
    volume: torch.Tensor
    cavity_label: torch.Tensor
    lca_label: torch.Tensor
    rca_label: torch.Tensor
    affine: np.ndarray
    lca_centering_affine: np.ndarray
    rca_centering_affine: np.ndarray
    
    def __post_init__(self):
        assert self.volume.ndim == 5 and self.volume.shape[0] == 1 and self.volume.shape[1] == 1, "Volume tensor must be in shape (1, 1, D, H, W)"
        assert self.cavity_label.shape == self.volume.shape, "Cavity label shape must match volume shape"
        assert self.lca_label.shape == self.volume.shape, "LCA label shape must match volume shape"
        assert self.rca_label.shape == self.volume.shape, "RCA label shape must match volume shape"
        assert self.cavity_label.dtype == torch.uint8, "Cavity label must be of type int8"
        assert self.lca_label.dtype == torch.bool, "LCA label must be of type bool"
        assert self.rca_label.dtype == torch.bool, "RCA label must be of type bool"
    
    def to_device(self, device: torch.device) -> "DataReaderResult":
        """Move all tensors to the specified device"""
        self.volume = self.volume.to(device)
        self.cavity_label = self.cavity_label.to(device)
        self.lca_label = self.lca_label.to(device)
        self.rca_label = self.rca_label.to(device)
        return self
    
    def get_coronary_central_line(
        self, 
        coronary_type: Literal["LCA", "RCA"],
        coordinate_system: Literal["world", "voxel", "coroanry_centering"] = "voxel"
    ) -> np.ndarray:
        """
        Get the central line of the coronary in world coordinate
        
        Args:
            coronary_type (Literal["LCA", "RCA"]): coronary type
            coordinate_system (Literal["world", "voxel", "coroanry_centering"]): coordinate system: 
                "world": world coordinate, xyz = nii_image_affine @ voxel_coordinate
                "voxel": voxel coordinate
                "coroanry_centering": coroanry centering coordinate that put the coroanry at the ordinate of the world coordinate
        
        Returns:
            central_line (np.ndarray): central line in world coordinate, shape: (N, 3)
        """
        if coronary_type == "LCA":
            label = self.lca_label
        elif coronary_type == "RCA":
            label = self.rca_label
        else:
            raise ValueError(f"Invalid coronary type: {coronary_type}")
        
        match coronary_type, coordinate_system:
            case _, "world":
                affine = self.affine
            case _, "voxel":
                affine = np.eye(4)
            case "LCA", "coroanry_centering":
                affine = self.lca_centering_affine
            case "RCA", "coroanry_centering":
                affine = self.rca_centering_affine
            case _:
                raise ValueError(f"Invalid coordinate system: {coordinate_system}")
        
        skel = skeletonize(label.squeeze().cpu().numpy())
        skel_xyz = np.nonzero(skel)
        skel_xyz = np.stack(skel_xyz, axis=1)
        skel_xyz1 = np.ones((skel_xyz.shape[0], 4))
        skel_xyz1[:, :3] = skel_xyz
        skel_xyz1 = (affine @ skel_xyz1.T).T
        skel_xyz1 = skel_xyz1[:, :3]
        return skel_xyz
    
    def save(
        self, 
        output_dir: Path,
        output_nii: bool = True,
        output_central_line: bool = True,
        central_line_coordinate_type: Literal["world", "voxel", "coroanry_centering"] = "voxel"
    ):
        """Save all tensors to the specified directory"""
        from ...saver import save_nii
        output_case_dir = output_dir / f"{self.phase}"
        output_case_dir.mkdir(exist_ok=True, parents=True)
        
        if output_nii:
            save_nii(output_case_dir / "volume.nii.gz", self.volume, self.affine)
            save_nii(output_case_dir / "cavity_label.nii.gz", self.cavity_label, self.affine, is_label=True)
            save_nii(output_case_dir / "lca_label.nii.gz", self.lca_label, self.affine, is_label=True)
            save_nii(output_case_dir / "rca_label.nii.gz", self.rca_label, self.affine, is_label=True)
        
        if output_central_line:
            lca_central_line = self.get_coronary_central_line("LCA", central_line_coordinate_type)
            rca_central_line = self.get_coronary_central_line("RCA", central_line_coordinate_type)
            np.save(output_case_dir / f"lca_central_line_{central_line_coordinate_type}.npy", lca_central_line)
            np.save(output_case_dir / f"rca_central_line_{central_line_coordinate_type}.npy", rca_central_line)


# TODO 用 Dataset/Dataloader 的形式实现 reader
class DataReader(Protocol):
    n_phases: int
    origin_image_size: tuple[int, int, int]
    origin_image_affine: np.ndarray
    lca_centering_affine: np.ndarray
    rca_centering_affine: np.ndarray

    def get_data(self, phase: CardiacPhase) -> DataReaderResult:
        ...