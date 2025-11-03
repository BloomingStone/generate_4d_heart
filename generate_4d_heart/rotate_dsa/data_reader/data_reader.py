from typing import Protocol, Literal
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import ndarray
from skimage.morphology import skeletonize
import cupy as cp
from cupyx.scipy.ndimage import label, center_of_mass, zoom
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.dlpack import from_dlpack as dlpack2tensor
from torch.utils.dlpack import to_dlpack as tensor2dlpack
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image
import pyvista as pv
import pyacvd

from ...roi import ROI
from ..cardiac_phase import CardiacPhase
from ..types import CoronaryType


def pre_load(file: Path) -> tuple[Nifti1Image, ndarray]:
    if not file.exists():
        raise FileNotFoundError(f"nii file not found: {file}")
    image_nii = nib_load(file)
    assert isinstance(image_nii, Nifti1Image)
    affine = image_nii.affine if image_nii.affine is not None else np.eye(4)
    return image_nii, affine

def load_nifti(file: Path, is_label: bool = False) -> tuple[Tensor, ndarray]:
    """load nifti file as torch tensor (shape: 1, 1, D, H, W) and return its affine matrix"""
    img, affine = pre_load(file)
    tensor = torch.from_numpy(img.get_fdata())
    if is_label:
        tensor = tensor.round().to(torch.uint8)
    else:
        tensor = tensor.to(torch.float32)
    tensor = tensor[None][None]  # add batch and channel dim
    return tensor, affine

def load_nifti_with_roi_crop(file: Path, roi: ROI, is_label: bool = False) -> tuple[Tensor, ndarray]:
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
) -> Tensor:
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


def separate_coronary(coronary: Tensor) -> tuple[Tensor, Tensor]:
    """
    separate coronary to LCA(1) and RCA(2)
    Args:
        coronary (Tensor): coronary segmentation
    Returns:
        lca (Tensor): LCA segmentation, dtype: torch.bool
        rca (Tensor): RCA segmentation, dtype: torch.bool
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
    original_affine: ndarray,
    center_voxel: tuple[int, int, int]
) -> ndarray:
    """get the affine that set the center of the image to the given center_voxel
    """
    B = original_affine[:3, :3]
    new_t = -(B @ np.array(center_voxel))
    T = np.eye(4)
    T[:3, :3] = B
    T[:3, 3] = new_t
    return T

def get_coronary_centering_affine(coronary_label: Tensor, volume_affine: ndarray) -> ndarray:
    """
    Compute an affine transform that recenters the heart coronary region in world coordinates for CTA

    The centroid of the coronary label (LCA or RCA) is computed in voxel space.
    We then build a new target center by combining the label centroid with the 
    geometric center of the image:
        - X (left–right) and Y (anterior–posterior) use the mean of image center and label centroid
        - Z (superior–inferior) uses the middle of the image (image center on Z axis)

    This heuristic shifts the heart region toward the center of the field of view 
    while preserving vertical alignment.

    Args:
        coronary_label (Tensor): Binary coronary artery mask of shape (1, 1, D, H, W).
        volume_affine (ndarray): Original volume affine matrix of shape (4, 4).

    Returns:
        ndarray: New affine matrix that centers the coronary structure in world space.
    """
    label_center: tuple[int, int, int] = center_of_mass(cp.from_dlpack(tensor2dlpack(coronary_label.squeeze().cuda()))) # type: ignore
    W, H, D = coronary_label.shape[-3:]
    image_center = (W/2, H/2, D/2)
    label_center_voxel = (
        int( (image_center[0] + label_center[0]) / 2 ), # left and right, set as the mean of image_center and label_center
        int( (image_center[1] + label_center[1]) / 2 ), # antero-posterior, same as above
        int( image_center[2] )                          # up and down, set as the center of image
    )
    
    return recenter(volume_affine, label_center_voxel)

def get_mesh_in_voxel(label: Tensor) -> pv.PolyData:
    label_np = label.squeeze().cpu().numpy().astype(np.uint8)
    label_big = zoom(cp.asarray(label_np), zoom=2).astype(cp.uint8).get()  # type: ignore
    mesh = pv.wrap(label_big)\
        .contour([1], method='flying_edges')\
        .smooth_taubin(
            n_iter=40, pass_band=0.001, normalize_coordinates=True)\
        .triangulate().clean()
    cluster = pyacvd.Clustering(mesh)
    cluster.cluster(10000)
    
    mesh: pv.PolyData = cluster.create_mesh().triangulate().clean()  # type: ignore
    mesh.points /= 2.0  # 因为上采样了2倍，所以点坐标要除以2
    return mesh

def get_mesh_in_world(label: Tensor, affine: ndarray) -> pv.PolyData:
    mesh = get_mesh_in_voxel(label)
    mesh.points = apply_affine(mesh.points, affine)
    return mesh

def apply_affine(points: Tensor | ndarray, affine: ndarray) -> ndarray:
    assert len(points.shape) == 2 and points.shape[-1] == 3
    if isinstance(points, ndarray):
        points = torch.from_numpy(points)
    _affine = torch.from_numpy(affine).to(device=points.device, dtype=points.dtype)
    new_points = F.pad(points, (0, 1), "constant", 1)   # shape=(N, 4), [x, y, z, 1]
    new_points = new_points @ _affine.T
    new_points = new_points[:, :3]
    return new_points.cpu().numpy()

    
@dataclass
class Coronary:
    type: CoronaryType
    label: Tensor
    centering_affine: ndarray
    
    mesh_in_world: pv.PolyData
    

@dataclass
class DataReaderResult:
    """ Holds the result of reading data for a specific phase. Labels are int8 tensors. tensors are in shape (B=1, C=1, D, H, W)."""
    phase: CardiacPhase
    
    volume: Tensor
    cavity_label: Tensor
    affine: ndarray
    
    coronary: Coronary
    
    def __post_init__(self):
        assert self.volume.ndim == 5 and self.volume.shape[0] == 1 and self.volume.shape[1] == 1, "Volume tensor must be in shape (1, 1, D, H, W)"
        assert self.cavity_label.shape == self.volume.shape, "Cavity label shape must match volume shape"
        assert self.coronary.label.shape == self.volume.shape, "Coronary label shape must match volume shape"
        assert self.cavity_label.dtype == torch.uint8, "Cavity label must be of type int8"
        assert self.coronary.label.dtype == torch.bool, "LCA label must be of type bool"
    
    def to_device(self, device: torch.device) -> "DataReaderResult":
        """Move all tensors to the specified device"""
        return DataReaderResult(
            self.phase,
            self.volume.to(device),
            self.cavity_label.to(device),
            self.affine, 
            Coronary(
                type=self.coronary.type,
                label=self.coronary.label.to(device),
                centering_affine=self.coronary.centering_affine,
                mesh_in_world=self.coronary.mesh_in_world,
            )
        )
    
    def get_coronary_central_line(
        self, 
        coordinate_system: Literal["world", "voxel", "coroanry_centering"] = "coroanry_centering"
    ) -> ndarray:
        """
        Get the central line of the coronary in world coordinate
        
        Args:
            coordinate_system (Literal["world", "voxel", "coroanry_centering"]): coordinate system: 
                "world": world coordinate, xyz = nii_volume_affine @ voxel_coordinate
                "voxel": voxel coordinate
                "coroanry_centering": coroanry centering coordinate that put the coroanry at the ordinate of the world coordinate
        
        Returns:
            central_line (ndarray): central line in world coordinate, shape: (N, 3)
        """
        match coordinate_system:
            case "world":
                affine = self.affine
            case "voxel":
                affine = np.eye(4)
            case "coroanry_centering":
                affine = self.coronary.centering_affine
            case _:
                raise ValueError(f"Invalid coordinate system: {coordinate_system}")
        
        skel = skeletonize(self.coronary.label.squeeze().cpu().numpy())
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
        output_mesh: bool = True,
        output_central_line: bool = True,
        central_line_coordinate_type: Literal["world", "voxel", "coroanry_centering"] = "coroanry_centering"
    ):
        """Save all tensors to the specified directory"""
        from ...saver import save_nii
        output_case_dir = output_dir / f"{self.phase}"
        output_case_dir.mkdir(exist_ok=True, parents=True)
        
        if output_nii:
            save_nii(output_case_dir / "volume.nii.gz", self.volume, self.affine)
            save_nii(output_case_dir / "cavity_label.nii.gz", self.cavity_label, self.affine, is_label=True)
            save_nii(output_case_dir / f"{self.coronary.type}_label.nii.gz", self.coronary.label, self.affine, is_label=True)
        
        if output_mesh:
            self.coronary.mesh_in_world.save(output_case_dir / f"{self.coronary.type}_mesh.vtk")
        
        if output_central_line:
            lca_central_line = self.get_coronary_central_line(central_line_coordinate_type)
            np.save(output_case_dir / f"{self.coronary.type}_central_line_{central_line_coordinate_type}.npy", lca_central_line)


# TODO 用 Dataset/Dataloader 的形式实现 reader
class DataReader(Protocol):
    n_phases: int
    _origin_volume_size: tuple[int, int, int]
    _origin_volume_affine: ndarray

    def get_data(self, phase: CardiacPhase, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        ...

    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        ...
    
    @property
    def lca_centering_affine(self) -> ndarray:
        ...
    
    @property
    def rca_centering_affine(self) -> ndarray:
        ...
    
    @property
    def volume_size(self) -> tuple[int, int, int]:
        return self._origin_volume_size
    
    @property
    def volume_affine(self) -> ndarray:
        return self._origin_volume_affine