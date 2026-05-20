from typing import Protocol, Literal
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy import ndarray
from skimage.morphology import skeletonize
import cupy as cp
from cupyx.scipy.ndimage import label, center_of_mass
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.dlpack import from_dlpack as dlpack2tensor
from torch.utils.dlpack import to_dlpack as tensor2dlpack   #type: ignore
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image
import pyvista as pv

from ...roi import ROI
from ..cardiac_phase import CardiacPhase
from ..types import CoronaryType
from ..contrast_simulator import ContrastSimulator


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
    device: torch.device
) -> Tensor:
    """
    read dvf from nifti file and zoom it to the roi size
    """
    dvf_nii, _ = pre_load(dvf_path)
    dvf_tensor = torch.from_numpy(dvf_nii.get_fdata()).to(device=device, dtype=torch.float, non_blocking=True)  # (H,W,D,1,3)
    
    zoom_rate = torch.from_numpy((1 / roi.get_zoom_rate()).flatten()).to(dvf_tensor, non_blocking=True)  # (3,)
    dvf_tensor.squeeze_().mul_(zoom_rate)  #(H, W, D, 3)
    dvf_tensor = F.interpolate(dvf_tensor, size=roi.get_roi_size_before_crop(), mode='trilinear', align_corners=False)  # (H', W', D', 3)
    dvf_tensor = dvf_tensor.squeeze().permute(3, 0, 1, 2)[None] # (1,3,H,W,D)
    return dvf_tensor


def separate_coronary(coronary: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
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
    
    with cp.cuda.Device(device.index):
        coronary = cp.from_dlpack(tensor2dlpack(coronary.to(device)))
        labeled_array, num_features = label(coronary.astype(cp.int8))  # type: ignore
        
        if num_features <= 1:
            raise ValueError("Coronary segmentation must have at least 2 components")
        
        assert labeled_array is not None
        component_sizes = cp.bincount(labeled_array.ravel())[1:]  # Skip background (0)
        
        largest_indices = cp.argsort(component_sizes)[-2:][::-1] + 1
        
        region_0 = (labeled_array == largest_indices[0]).astype(cp.bool_)
        region_1 = (labeled_array == largest_indices[1]).astype(cp.bool_)
        
        center_0 = center_of_mass(region_0.astype(cp.float16))  # use float16 for fast computation
        center_1 = center_of_mass(region_1.astype(cp.float16))
        
        region_0 = dlpack2tensor(region_0).reshape(shape).to(torch.bool)
        region_1 = dlpack2tensor(region_1).reshape(shape).to(torch.bool)
    
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

import cupy as cp

def get_coronary_centering_affine(coronary_label: Tensor, volume_affine: ndarray, device: torch.device) -> ndarray:
    W, H, D = coronary_label.shape[-3:]
    image_center = (W//2, H//2, D//2)
    with cp.cuda.Device(device.index):
        # 1. 将数据转为 cupy 阵列并压缩维度
        label_cp = cp.from_dlpack(tensor2dlpack(coronary_label.squeeze().to(device)))
        
        # 2. 找到所有非零（标签）点的坐标
        coords = cp.argwhere(label_cp > 0)
        
        if coords.size == 0:
            # 如果没找到标签，退回到图像物理中心
            label_center_voxel = image_center
        else:
            # 3. 计算 AABB (Axis-Aligned Bounding Box) 的边界
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            
            # 4. 计算包围盒的中心
            # 这能确保最左边和最右边的分支被平等对待
            label_center_voxel = (
                int((min_coords[0] + max_coords[0]) / 2),
                int((min_coords[1] + max_coords[1]) / 2),
                int(image_center[2])    # Z轴保持在图像中心
            )

        return recenter(volume_affine, label_center_voxel)

def get_mesh_in_voxel(label: Tensor,max_points: int|None=10000) -> pv.PolyData:
    label_np = label.squeeze().cpu().numpy()
    label_np = (label_np>0.5).astype(np.uint8)
    mesh: pv.PolyData = pv.wrap(label_np)\
        .contour([1], method='flying_edges')\
        .triangulate().smooth_taubin().clean()
    
    if max_points is not None and mesh.n_points > max_points:
        decimate_rate = max_points / mesh.n_points
        mesh = mesh.decimate_pro(reduction=decimate_rate, preserve_topology=True, feature_angle=30.0)  #type: ignore
    return mesh

def get_mesh_in_world(label: Tensor, affine: ndarray, max_points: int|None=10000) -> pv.PolyData:
    mesh = get_mesh_in_voxel(label, max_points)
    mesh.points = apply_affine(mesh.points, affine)
    return mesh

def apply_affine(points: Tensor | ndarray, affine: ndarray) -> ndarray:
    assert points.shape[-1] == 3
    if isinstance(points, ndarray):
        points = torch.from_numpy(points)
    points = points.to(torch.float32)
    _affine = torch.from_numpy(affine).to(device=points.device, dtype=points.dtype)
    new_points = F.pad(points, (0, 1), "constant", 1)   # shape=(..., 4), [x, y, z, 1]
    new_points = new_points @ _affine.T
    new_points = new_points[..., :3]
    return new_points.cpu().numpy()

    
@dataclass
class Coronary:
    type: CoronaryType
    
    # Different from original volume as the contrast simulator may modify the volume differently for LCA and RCA
    volume: Tensor 
    
    label: Tensor
    
    original_affine: ndarray
    centering_affine: ndarray
    
    # Its points should be matched with the label after corresponding affine. 
    # i.e. mesh_centering.points = centering_affine @ voxel_coordinate of the label
    # mesh_original.points = original_affine @ voxel_coordinate of the label
    mesh_original: pv.PolyData
    mesh_centering: pv.PolyData = field(init=False)
    
    def __post_init__(self):
        self.mesh_centering = self.mesh_original.copy()
        ori_T = self.original_affine[:3, 3]
        centering_T = self.centering_affine[:3, 3]
        translation = centering_T - ori_T
        self.mesh_centering.points += translation
    
    
    def to_device(self, device: torch.device) -> "Coronary":
        return Coronary(
            type=self.type,
            volume=self.volume.to(device),
            label=self.label.to(device),
            original_affine=self.original_affine,
            centering_affine=self.centering_affine,
            mesh_original=self.mesh_original,
        )
    
    @property
    def mesh_in_voxel(self) -> pv.PolyData:
        inv_affine = np.linalg.inv(self.original_affine)
        mesh = self.mesh_original.copy()
        mesh = mesh.transform(inv_affine, inplace=False)
        return mesh # type: ignore
    
    def get_coronary_central_line(
        self, 
        coordinate_system: Literal["world", "voxel", "coronary_centering"] = "coronary_centering"
    ) -> ndarray:
        """
        Get the central line of the coronary in world coordinate
        
        Args:
            coordinate_system (Literal["world", "voxel", "coronary_centering"]): coordinate system: 
                "world": world coordinate, xyz = nii_volume_affine @ voxel_coordinate
                "voxel": voxel coordinate
                "coronary_centering": coronary centering coordinate that put the coronary at the ordinate of the world coordinate
        
        Returns:
            central_line (ndarray): central line in world coordinate, shape: (N, 3)
        """
        match coordinate_system:
            case "world":
                affine = self.original_affine
            case "voxel":
                affine = np.eye(4)
            case "coronary_centering":
                affine = self.centering_affine
            case _:
                raise ValueError(f"Invalid coordinate system: {coordinate_system}")
        
        skel = skeletonize(self.label.squeeze().cpu().numpy())
        skel_xyz = np.stack(np.nonzero(skel), axis=1)
        skel_xyz = apply_affine(skel_xyz, affine)
        return skel_xyz
    
    

@dataclass
class DataReaderResult:
    """ Holds the result of reading data for a specific phase. Labels are int8 tensors. tensors are in shape (B=1, C=1, D, H, W)."""
    phase: CardiacPhase
    cavity_label: Tensor
    affine: ndarray
    
    coronary: Coronary
    
    def __post_init__(self):
        # coronary.volume is the canonical volume; validate shapes against it
        assert self.coronary.volume.ndim == 5 and self.coronary.volume.shape[0] == 1 and self.coronary.volume.shape[1] == 1, "Coronary.volume tensor must be in shape (1, 1, D, H, W)"
        assert self.cavity_label.shape == self.coronary.volume.shape, "Cavity label shape must match coronary.volume shape"
        assert self.coronary.label.shape == self.coronary.volume.shape, "Coronary label shape must match coronary.volume shape"
        assert self.cavity_label.dtype == torch.uint8, "Cavity label must be of type int8"
        assert self.coronary.label.dtype == torch.bool, "LCA label must be of type bool"
    
    def to_device(self, device: torch.device) -> "DataReaderResult":
        """Move all tensors to the specified device"""
        return DataReaderResult(
            self.phase,
            self.cavity_label.to(device),
            self.affine, 
            self.coronary.to_device(device)
        )
    
    def save(
        self, 
        output_dir: Path,
        output_nii: bool = True,
        output_mesh: bool = True,
        output_central_line: bool = True,
        coordinate_system: Literal["world", "voxel", "coronary_centering"] = "coronary_centering"
    ):
        """Save all tensors to the specified directory"""
        from ...saver import save_nii
        output_case_dir = output_dir / f"{self.phase}"
        output_case_dir.mkdir(exist_ok=True, parents=True)
        
        if output_nii:
            save_nii(output_case_dir / "cavity_label.nii.gz", self.cavity_label, self.affine, is_label=True)
            save_nii(output_case_dir / f"{self.coronary.type}_volume.nii.gz", self.coronary.volume, self.affine)
            save_nii(output_case_dir / f"{self.coronary.type}_label.nii.gz", self.coronary.label, self.affine, is_label=True)
        
        if output_mesh:
            match coordinate_system:
                case "world":
                    mesh = self.coronary.mesh_original
                case "voxel":
                    mesh = self.coronary.mesh_in_voxel
                case "coronary_centering":
                    mesh = self.coronary.mesh_centering
                case _:
                    raise ValueError(f"Invalid coordinate system: {coordinate_system}")
            mesh.save(output_case_dir / f"{self.coronary.type}_mesh_{coordinate_system}.vtp", binary=True)
        
        if output_central_line:
            lca_central_line = self.coronary.get_coronary_central_line(coordinate_system)
            np.savez_compressed(output_case_dir / f"{self.coronary.type}_central_line_{coordinate_system}.npz", lca_central_line)


class DataReader(Protocol):
    
    @dataclass
    class _Data:
        affine: np.ndarray
        cavity: Tensor
        coronary: dict[CoronaryType, Coronary]
        
        def __post_init__(self):
            assert self.cavity.shape == next(iter(self.coronary.values())).volume.shape, "Cavity shape must match coronary volume shape"
            assert np.allclose(self.coronary[CoronaryType.LCA].original_affine, self.coronary[CoronaryType.RCA].original_affine, self.affine), "LCA and RCA must have the same original affine as the cavity affine"
        
        @staticmethod
        def init_from_volume(
            volume: Tensor,
            cavity: Tensor,
            coronary: Tensor,
            affine: np.ndarray,
            running_device: torch.device,
            contrast_simulator: ContrastSimulator|None
        ) -> "DataReader._Data":
            lca, rca = separate_coronary(coronary, running_device)
            if contrast_simulator is not None:
                print("Preprocessing baseline volume with contrast simulator...")
                volume = contrast_simulator.preprocess(volume, cavity)
            lca_volume = volume.clone()
            rca_volume = volume.clone()
            
            if contrast_simulator is not None and not contrast_simulator.contrast_change_over_time:
                print("Applying STATIC contrast simulator to original volume...")
                # simulate focuses on coronary-only modification
                lca_volume = contrast_simulator.simulate(lca_volume, cavity, lca)
                rca_volume = contrast_simulator.simulate(rca_volume, cavity, rca)
            
            return DataReader._Data(
                affine=affine,
                cavity=cavity,
                coronary={
                    CoronaryType.LCA: Coronary(
                        type=CoronaryType.LCA,
                        volume=lca_volume,
                        label=lca,
                        original_affine=affine,
                        centering_affine=get_coronary_centering_affine(lca, affine, running_device),
                        mesh_original=get_mesh_in_world(lca, affine)
                    ),
                    CoronaryType.RCA: Coronary(
                        type=CoronaryType.RCA,
                        volume=rca_volume,
                        label=rca,
                        original_affine=affine,
                        centering_affine=get_coronary_centering_affine(rca, affine, running_device),
                        mesh_original=get_mesh_in_world(rca, affine)
                    )
                }
            )
        
        def __getitem__(self, key: CoronaryType) -> Coronary:
            return self.coronary[key]
        
        def crop_by_roi(self, roi: ROI) -> "DataReader._Data":
            def crop(x: Tensor) -> Tensor:
                return roi.crop_on_data(x.clone())

            cropped_affine = roi.affine_after_crop
            coronary_old = self.coronary
            
            coronary_new = {}
            for cor_type, cor in coronary_old.items():
                orinial_T = cor.original_affine[:3, 3]
                centering_T = cor.centering_affine[:3, 3]
                translation = centering_T - orinial_T
                cropped_centering_T = translation + cropped_affine[:3, 3]
                cropped_centering_affine = cropped_affine.copy()
                cropped_centering_affine[:3, 3] = cropped_centering_T
                coronary_new[cor_type] = Coronary(
                    cor_type, crop(cor.volume), crop(cor.label),
                    original_affine=cropped_affine,
                    centering_affine=cropped_centering_affine,
                    mesh_original=cor.mesh_original.copy()  # mesh is not changed by cropping
                )
            
            return DataReader._Data(
                affine=cropped_affine,
                cavity=crop(self.cavity),
                coronary=coronary_new
            )
            
     
        @property
        def all_coronary_label(self) -> Tensor:
            res = self[CoronaryType.LCA].label.clone().to(torch.uint8)
            res += self[CoronaryType.RCA].label
            return res
        
        @property
        def lca(self) -> Coronary:
            return self.coronary[CoronaryType.LCA]
        
        @property
        def rca(self) -> Coronary:
            return self.coronary[CoronaryType.RCA]
        
    # end of _Data class
    
    n_phases: int
    
    # if contrast_simulator.contrast_change_over_time == True, the contrast simulator should be called in `self.get_data` with `global_time` input, 
    # otherwise, the contrast simulator should be called in initialization and the `global_time` in `self.get_data` should be None and ignored.
    # All readers expose a contrast_simulator attribute. Readers that don't simulate
    # contrast should set it to `IdentityContrast()` by default.
    contrast_simulator: ContrastSimulator

    _origin_volume_size: tuple[int, int, int]
    _origin_volume_affine: ndarray

    def get_data(
        self, 
        phase: CardiacPhase, 
        coronary_type: CoronaryType | Literal["LCA", "RCA"],
        global_time: float=0.0
    ) -> DataReaderResult:
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


 