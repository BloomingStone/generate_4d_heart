from typing import Protocol, Literal, Any
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
from ...med_image import MedImage, Original, Centering, CoordinateSystem
from ..cardiac_phase import CardiacPhase
from ...types import CoronaryType
from ..contrast_simulator import ContrastSimulator
from ..artifacts import ArtifactManager, Artifact


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
    dvf_tensor = dvf_tensor.permute(3, 0, 1, 2)[None] # (1,3,H,W,D)
    dvf_tensor = F.interpolate(dvf_tensor, size=roi.get_roi_size_before_crop(), mode='trilinear', align_corners=False)  # (1,3, H', W', D')
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


CenteringMode = Literal["coronary_center", "image_center"]

def get_centering_pos_in_origin_world(
    coronary_label: Tensor,
    volume_affine: ndarray,
    device: torch.device,
    centering_mode: CenteringMode = "coronary_center"
) -> ndarray:
    """
    Args:
        coronary_label: coronary label tensor.
        volume_affine: affine matrix of the original volume.
        device: torch device.
        centering_mode: how to determine the target center voxel.
            - "coronary_center" (default): use the AABB center of the coronary label.
            - "image_center": use the geometric center of the image volume shape.

    Returns:
        The positon of centring point in original world coordinate (mm), shape=(3,).
    """
    W, H, D = coronary_label.shape[-3:]
    image_center = (W // 2, H // 2, D // 2)

    B = volume_affine[:3, :3]
    T = volume_affine[:3, 3]
    if centering_mode == "image_center":
        return B @ np.array(image_center) + T

    # centering_mode == "coronary_center"
    with cp.cuda.Device(device.index):
        label_cp = cp.from_dlpack(tensor2dlpack(coronary_label.squeeze().to(device)))
        coords = cp.argwhere(label_cp > 0)

        if coords.size == 0:
            label_center_voxel = image_center
        else:
            # 3. AABB (Axis-Aligned Bounding Box) 的边界
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            
            # 计算包围盒的中心
            # 这能确保最左边和最右边的分支被平等对待
            label_center_voxel = (
                int((min_coords[0] + max_coords[0]) / 2),
                int((min_coords[1] + max_coords[1]) / 2),
                int(image_center[2])    # Z轴保持在图像中心
            )

        return B @ np.array(label_center_voxel) + T

def get_mesh_in_voxel(label: Tensor,max_points: int|None=10000) -> pv.PolyData:
    label_np = label.squeeze().cpu().numpy()
    label_np = (label_np>0.5).astype(np.uint8)
    mesh: pv.PolyData = pv.wrap(label_np)\
        .contour([1], method='flying_edges')\
        .triangulate().smooth_taubin().clean().triangulate()  # type: ignore
    
    if max_points is not None and mesh.n_points > max_points:
        decimate_rate = max_points / mesh.n_points
        try:
            mesh = mesh.decimate_pro(reduction=decimate_rate, preserve_topology=True, feature_angle=30.0)  #type: ignore
        except:
            print(f"Warning: mesh decimation failed, returning original mesh with {mesh.n_points} points (max_points={max_points})")
            pass
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
    """冠脉数据，用 MedImage 统一管理 volume/label 及其坐标系。

    Fields:
        type: LCA or RCA
        volume: MedImage[World] — 带对比剂的体积（含 affine = voxel→world）
        label: MedImage[World] — 冠脉标签（与 volume 共享同一 affine）
        centering_coord: Centering — 中心化坐标系，volume/label/mesh/artifacts 共享
        mesh_original: 世界坐标系下的冠脉网格
        artifacts_original: 世界坐标系下的伪影
    """
    type: CoronaryType

    volume: MedImage[Original]
    """ volume with contrast, dtype: torch.float32, shape: (1, 1, D, H, W) 
    Different from original volume as the contrast simulator may modify the volume differently for LCA and RCA"""

    label: MedImage[Original]
    """ Coronary label, dtype: torch.bool, shape: (1, 1, D, H, W) """

    centering_coord: Centering
    """ 中心化坐标系，与 volume.affine 共享 B 矩阵，仅平移不同。 """

    mesh_original: pv.PolyData
    """ coronary mesh in original affine coordinate (world coordinate) """

    artifacts_original: list[Artifact] | None = None
    """ Artifacts in original affine coordinate (world coordinate) """

    # ── 计算属性 ──

    @property
    def original_affine(self) -> ndarray:
        return self.volume.affine

    @property
    def centering_affine(self) -> ndarray:
        return self.volume_centering.affine

    @property
    def T_original_to_centering(self) -> ndarray:
        """ Translation vector from original affine to centering affine. """
        return -self.centering_coord.translation

    def to_device(self, device: torch.device) -> "Coronary":
        return Coronary(
            type=self.type,
            volume=self.volume.to(device),
            label=self.label.to(device),
            centering_coord=self.centering_coord,
            mesh_original=self.mesh_original,
            artifacts_original=self.artifacts_original,
        )

    @property
    def volume_centering(self) -> MedImage[Centering]:
        return self.volume.to_coord(self.centering_coord)

    @property
    def label_centering(self) -> MedImage[Centering]:
        return self.label.to_coord(self.centering_coord)

    @property
    def mesh_in_voxel(self) -> pv.PolyData:
        inv_affine = np.linalg.inv(self.original_affine)
        mesh = self.mesh_original.copy()
        mesh = mesh.transform(inv_affine, inplace=False)
        return mesh # type: ignore

    def get_coronary_central_line(
        self,
        coordinate_system: Literal["world", "voxel", "centering"] = "centering"
    ) -> ndarray:
        match coordinate_system:
            case "world":
                img = self.volume
            case "voxel":
                img = MedImage(self.label.data, np.eye(4), Original())
            case "centering":
                img = self.label_centering
            case _:
                raise ValueError(f"Invalid coordinate system: {coordinate_system}")

        skel = skeletonize(self.label.data.squeeze().cpu().numpy())
        skel_xyz = np.stack(np.nonzero(skel), axis=1)
        skel_xyz = apply_affine(skel_xyz, img.affine)
        return skel_xyz

    @property
    def mesh_centering(self) -> pv.PolyData:
        """ coronary mesh in centering coordinate """
        return self.mesh_original.copy().translate(
            self.T_original_to_centering, inplace=False
        )  # type: ignore

    @property
    def artifacts_centering(self) -> list[Artifact] | None:
        """ artifacts in centering coordinate """
        if self.artifacts_original is None:
            return None
        return [a.translate(self.T_original_to_centering) for a in self.artifacts_original]
    
@dataclass
class DataReaderResult:
    """ Holds the result of reading data for a specific phase.

    cavity: MedImage[World] — 心腔标签
    coronary: Coronary — 冠脉数据（内含 MedImage）
    """
    phase: CardiacPhase
    cavity: MedImage[Original]
    coronary: Coronary

    def __post_init__(self):
        vol = self.coronary.volume
        cav = self.cavity
        assert vol.data.ndim == 5 and vol.data.shape[0] == 1 and vol.data.shape[1] == 1
        assert cav.data.shape == vol.data.shape
        assert self.coronary.label.data.shape == vol.data.shape
        assert cav.data.dtype == torch.uint8
        assert self.coronary.label.data.dtype == torch.bool
        assert np.allclose(self.coronary.original_affine, cav.affine)

    def to_device(self, device: torch.device) -> "DataReaderResult":
        return DataReaderResult(
            self.phase,
            self.cavity.to(device),
            self.coronary.to_device(device)
        )

    def save(
        self,
        output_dir: Path,
        *,
        save_nii_type: list[Literal["cavity", "volume", "coronary"]] | None = None,
        save_mesh: bool = False,
        save_central_line: bool = False,
        save_meta: bool = False,
        coordinate_system: Literal["world", "voxel", "centering"] = "centering"
    ) -> dict[str, Any]:
        """
        Save data to the specified directory
        
        Args:
            output_dir (Path): directory to save the data
            save_nii_type (list[Literal["cavity", "volume", "coronary"]]|None): which nifti files to save, if None, no nifti will be saved
            save_mesh (bool): whether to save coronary mesh as .vtp file
            save_central_line (bool): whether to save coronary central line as .npz file
            save_meta (bool): whether to save meta information as json file
            coordinate_system (Literal["world", "voxel", "centering"]): coordinate system for saved data
                - "world": world coordinate, xyz = nii_volume_affine @ voxel_coordinate, affine will be the original affine of the nii volume
                - "voxel": voxel coordinate, affine will be identity matrix
                - "centering": coronary centering coordinate that put the coronary at the ordinate of the world coordinate, 
                affine will be the centering affine of the coronary
        Returns:
            meta (dict): meta information of saving, includes phase, coronary type, and coordinate system
        """
        from ...saver import save_nii
        import json

        meta = {
            "phase": self.phase.to_str(precision=4),
            "coronary_type": self.coronary.type.value,
            "coordinate_system": coordinate_system,
        }

        match coordinate_system:
            case "world":
                cavity_img = self.cavity
                vol_img = self.coronary.volume
                label_img = self.coronary.label
                mesh = self.coronary.mesh_original
            case "voxel":
                cavity_img = MedImage(self.cavity.data, np.eye(4), Original())
                vol_img = MedImage(self.coronary.volume.data, np.eye(4), Original())
                label_img = MedImage(self.coronary.label.data, np.eye(4), Original())
                mesh = self.coronary.mesh_in_voxel
            case "centering":
                cavity_img = self.cavity.to_coord(self.coronary.centering_coord)
                vol_img = self.coronary.volume_centering
                label_img = self.coronary.label_centering
                mesh = self.coronary.mesh_centering
            case _:
                raise ValueError(f"Invalid coordinate system: {coordinate_system}")

        if save_meta:
            with open(output_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        if save_nii_type:
            if "cavity" in save_nii_type:
                save_nii(output_dir / "cavity_label.nii.gz", cavity_img.data, cavity_img.affine, is_label=True)
            if "volume" in save_nii_type:
                save_nii(output_dir / "volume_with_contrast.nii.gz", vol_img.data, vol_img.affine)
            if "coronary" in save_nii_type:
                save_nii(output_dir / "coronary_label.nii.gz", label_img.data, label_img.affine, is_label=True)

        if save_mesh:
            mesh.save(output_dir / "mesh.vtp", binary=True)

        if save_central_line:
            cl = self.coronary.get_coronary_central_line(coordinate_system)
            np.savez_compressed(output_dir / "central_line.npz", cl)

        return meta


class DataReader(Protocol):
    
    @dataclass
    class _Data:
        cavity: MedImage[Original]
        coronary: dict[CoronaryType, Coronary]

        def __post_init__(self):
            vol_shape = next(iter(self.coronary.values())).volume.data.shape
            assert self.cavity.data.shape == vol_shape, \
                f"Cavity shape {self.cavity.data.shape} != coronary volume shape {vol_shape}"
            assert np.allclose(
                self.coronary[CoronaryType.LCA].original_affine,
                self.coronary[CoronaryType.RCA].original_affine,
                self.cavity.affine
            ), "LCA and RCA must have the same original affine as the cavity affine"

        @staticmethod
        def init_from_volume(
            volume: Tensor,
            cavity: Tensor,
            coronary: Tensor,
            affine: np.ndarray,
            running_device: torch.device,
            contrast_simulator: ContrastSimulator | None,
            artifact_manager: ArtifactManager | None = None,
            centering_mode: CenteringMode = "coronary_center"
        ) -> "DataReader._Data":
            lca, rca = separate_coronary(coronary, running_device)
            if contrast_simulator is not None:
                print("Preprocessing baseline volume with contrast simulator...")
                volume = contrast_simulator.preprocess(volume, cavity, affine)

            lca_volume = volume.clone()
            rca_volume = volume.clone()

            if contrast_simulator is not None and not contrast_simulator.contrast_change_over_time:
                print("Applying STATIC contrast simulator to original volume...")
                lca_volume = contrast_simulator.simulate(lca_volume, cavity, lca, affine)
                rca_volume = contrast_simulator.simulate(rca_volume, cavity, rca, affine)

            def _build_coronary(cor_t: CoronaryType, cor_vol: Tensor, cor_label: Tensor) -> Coronary:
                return Coronary(
                    type=cor_t,
                    volume=MedImage(data=cor_vol, affine=affine, coord=Original()),
                    label=MedImage(data=cor_label, affine=affine, coord=Original()),
                    centering_coord=Centering(cor_t, get_centering_pos_in_origin_world(
                        cor_label, affine, running_device, centering_mode
                    )),
                    mesh_original=get_mesh_in_world(cor_label, affine),
                    artifacts_original=artifact_manager.generate_or_load(
                        coronary_label=cor_label.squeeze().cpu().numpy().astype(bool),
                        cavity_label=cavity.squeeze().cpu().numpy(),
                        affine=affine
                    ) if artifact_manager is not None else None
                )

            return DataReader._Data(
                cavity=MedImage(data=cavity, affine=affine, coord=Original()),
                coronary={
                    CoronaryType.LCA: _build_coronary(CoronaryType.LCA, lca_volume, lca),
                    CoronaryType.RCA: _build_coronary(CoronaryType.RCA, rca_volume, rca),
                }
            )

        def __getitem__(self, key: CoronaryType) -> Coronary:
            return self.coronary[key]

        def crop_by_roi(self, roi: ROI) -> "DataReader._Data":
            cropped_affine = roi.affine_after_crop
            cropped_cavity = MedImage(
                data=roi.crop_on_data(self.cavity.data.clone()),
                affine=cropped_affine,
                coord=Original(),
            )

            coronary_new = {}
            for cor_type, cor in self.coronary.items():
                # 裁剪后 B 矩阵变了，但 centering translation 不变，仍然是原始坐标系到中心化坐标系的平移向量
                coronary_new[cor_type] = Coronary(
                    type=cor_type,
                    volume=MedImage(
                        data=roi.crop_on_data(cor.volume.data.clone()),
                        affine=cropped_affine,
                        coord=Original(),
                    ),
                    label=MedImage(
                        data=roi.crop_on_data(cor.label.data.clone()),
                        affine=cropped_affine,
                        coord=Original(),
                    ),
                    centering_coord=cor.centering_coord,
                    mesh_original=cor.mesh_original.copy(),
                    artifacts_original=cor.artifacts_original,
                )

            return DataReader._Data(cavity=cropped_cavity, coronary=coronary_new)

        @property
        def all_coronary_label(self) -> Tensor:
            res = self[CoronaryType.LCA].label.data.clone().to(torch.uint8)
            res += self[CoronaryType.RCA].label.data
            return res

        @property
        def lca(self) -> Coronary:
            return self.coronary[CoronaryType.LCA]

        @property
        def rca(self) -> Coronary:
            return self.coronary[CoronaryType.RCA]

        def to_device(self, device: torch.device) -> "DataReader._Data":
            return DataReader._Data(
                cavity=self.cavity.to(device),
                coronary={cor_type: cor.to_device(device) for cor_type, cor in self.coronary.items()}
            )
    # end of _Data class
    
    n_phases: int
    
    contrast_simulator: ContrastSimulator
    """
    if contrast_simulator.contrast_change_over_time == True, the contrast simulator should be called in `self.get_data` with `global_time` input, 
    otherwise, the contrast simulator should be called in initialization and the `global_time` in `self.get_data` should be None and ignored.
    All readers expose a contrast_simulator attribute. Readers that don't simulate
    contrast should set it to `IdentityContrast()` by default.
    """
    
    artifact_manager: ArtifactManager|None = None
    
    centering_mode: CenteringMode = "coronary_center"
    """
    Determines how the centering affine is computed:
        - "coronary_center": center based on the AABB of the coronary label (default).
        - "image_center": center based on the geometric center of the image volume shape.
    """
    

    _origin_volume_size: tuple[int, int, int]
    _origin_volume_affine: ndarray

    def get_data(
        self, 
        phase: CardiacPhase, 
        coronary_type: CoronaryType | Literal["LCA", "RCA"],
        global_time: float=0.0
    ) -> DataReaderResult:
        ...

    def get_static_data(
        self,
        coronary_type: CoronaryType | Literal["LCA", "RCA"],
    ) -> list[tuple[MedImage[Centering], np.ndarray]]:
        """
        Return list of (MedImage[Centering], roi_aabb) for layered batch DRR rendering.
        roi_aabb 标记该层占用的体素包围盒.
        """
        return []

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


 