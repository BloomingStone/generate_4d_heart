from pathlib import Path
from typing import Type, Callable, override, Literal
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from monai.networks.blocks.warp import DVF2DDF, Warp
from tqdm import tqdm
import einops
import pyvista as pv

from ... import NUM_TOTAL_PHASE
from ...roi import ROI
from ..movement_enhancer import MovementEnhancer
from ..types import CoronaryType
from ..cardiac_phase import CardiacPhase
from .data_reader import (
    DataReader, DataReaderResult, Coronary, separate_coronary, get_coronary_centering_affine,
    load_and_zoom_dvf, load_nifti, get_mesh_in_voxel, apply_affine, get_mesh_in_world
)


@dataclass
class _Data:
    device: torch.device
    image: Tensor
    cavity: Tensor
    affine: np.ndarray
    lca: Tensor
    rca: Tensor
    lca_centering_affine: np.ndarray = field(init=False)
    rca_centering_affine: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.lca_centering_affine = get_coronary_centering_affine(self.lca, self.affine, self.device)
        self.rca_centering_affine = get_coronary_centering_affine(self.rca, self.affine, self.device)
    
    @property
    def all_coronary_label(self) -> Tensor:
        res = self.lca.clone().to(torch.uint8)
        res += self.rca
        return res


class LazyCardiacDVFWarpModule(nn.Module):
    """
    Compute warped image/labels from DVFs stored on CPU.
    Uses async transfer and small GPU cache for efficiency.
    """

    def __init__(
        self,
        data: _Data,
        device: torch.device,
        *,
        precomputed_ddf: bool,
        dvf_list: list[Tensor] | None = None,   #shape=(1, 3, D, H, W)
        ddf_list: list[Tensor] | None = None    #shape=(2, 3, D, H, W)
    ):
        super().__init__()

        # DVF data stored on CPU pinned memory
        self.data = data
        self.device = device
        self.dvf2ddf = DVF2DDF(num_steps=5, mode="bilinear", padding_mode="zeros").to(device).half()
        self.warp_image = Warp(mode="bilinear", padding_mode="zeros").to(device).half()
        self.warp_label = Warp(mode="nearest", padding_mode="zeros").to(device).half()
        self.n_phases = NUM_TOTAL_PHASE     # TODO for now, we only supported total 20 phases

        spatial_dims = 3
        self.spatial_size = torch.tensor(self.data.image.shape[-spatial_dims:]).to(device, non_blocking=True)   # (D H W)

        self._gpu_cache_max = 2
        self.gpu_cache = OrderedDict[int, Tensor]()

        # precomputed DDF support
        self.precomputed_ddf = precomputed_ddf
        self.ddf_list_cpu = ddf_list
        self.dvf_list_cpu = dvf_list

        self.image = data.image.half().to(device, non_blocking=True)
        self.cavity = data.cavity.half().to(device, non_blocking=True)
        
        # shape = (1, C=2, D H W), c=0: lca, c=1: rca
        self.coronary = torch.concat((data.lca, data.rca), dim=1).half().to(device, non_blocking=True)
        
        def init_branch(branch: Tensor) -> tuple[pv.PolyData, Tensor, Tensor]:
            mesh_voxel = get_mesh_in_voxel(branch)
            points = torch.from_numpy(mesh_voxel.points).half().to(device, non_blocking=True)
            points_norm = points / (self.spatial_size - 1) * 2 - 1   # Norm to [-1, 1]
            points_norm = einops.rearrange(points_norm, "n d -> 1 n 1 1 d")
            index_ordering: list[int] = list(range(spatial_dims - 1, -1, -1))
            points_norm = points_norm[..., index_ordering]  # z, y, x -> x, y, z
            return mesh_voxel, points, points_norm

        self.lca_mesh_voxel, self.lca_mesh_points, self.lca_mesh_points_norm = init_branch(data.lca)
        self.rca_mesh_voxel, self.rca_mesh_points, self.rca_mesh_points_norm = init_branch(data.rca)


    def _get_maybe_cached_x(self, idx: int) -> Tensor:
        """
        Load DVF to GPU if not cached. Uses async copy.
        """
        if idx in self.gpu_cache:
            # move this key to the end to mark as recently used
            x = self.gpu_cache.pop(idx)
            self.gpu_cache[idx] = x
            return x

        if self.precomputed_ddf and self.ddf_list_cpu is not None:
            x = self.ddf_list_cpu[idx].to(self.device)
        elif (not self.precomputed_ddf) and (self.dvf_list_cpu is not None):
            x = self.dvf_list_cpu[idx].to(self.device)
        else:
            raise RuntimeError("No DVF or DDF data available.")

        # evict least-recently-used entries only when we will exceed the capacity
        while len(self.gpu_cache) >= self._gpu_cache_max:
            _, x_old = self.gpu_cache.popitem(last=False)
            del x_old

        self.gpu_cache[idx] = x
        return x


    def _points_warp_and_to_world(self, ddf_inverse: Tensor, coronary_type: CoronaryType) -> pv.PolyData:
        if coronary_type == CoronaryType.LCA:
            points = self.lca_mesh_points
            points_norm = self.lca_mesh_points_norm
            mesh: pv.PolyData = self.lca_mesh_voxel.copy()
            centering_affine = self.data.lca_centering_affine
        else:
            points = self.rca_mesh_points
            points_norm = self.rca_mesh_points_norm
            mesh: pv.PolyData = self.rca_mesh_voxel.copy()
            centering_affine = self.data.rca_centering_affine
        
        delta = F.grid_sample(
            ddf_inverse,    # (1, 3, h, w, d)
            points_norm,         # (1, n, 1, 1, 3)
            mode='bilinear',
            align_corners=True
        )  # (1, 3, n, 1, 1)

        # times centering affine to match label: tansform coronary to the origin of world coordiantes
        new_points = apply_affine(
            points + delta.squeeze().T,   # (n, 3)
            centering_affine
        )
        mesh.points = new_points.astype(np.float32)
        return mesh

    @torch.inference_mode()
    def forward(self, phase: CardiacPhase, coronary_type: CoronaryType) -> tuple[
        Tensor, Tensor, Tensor, pv.PolyData
    ]:
        """
        Compute warped tensors for the given phase ∈ [0,1)
        """
        idx0 = phase.closest_index_floor(self.n_phases)
        idx1 = phase.closest_index_ceil(self.n_phases)
        w = float(phase) * self.n_phases - idx0

        # --- async copy from CPU ---
        x = self._get_maybe_cached_x(idx0)     # shape = (2, C = 3, W, H, D)
        x1 = self._get_maybe_cached_x(idx1)
        with torch.no_grad():
            x = (x * (1 - w) + x1 * w)
            if self.precomputed_ddf:
                assert x.shape[0] == 2
            else:
                assert x.shape[0] == 1
                x: Tensor = self.dvf2ddf(torch.cat((x, -x), dim=0))
        ddf = x[0:1]
        ddf_inv = x[1:2]

        img_warped: Tensor = self.warp_image(self.image, ddf)
        coronary = self.coronary[:, 0:1] if coronary_type == CoronaryType.LCA else self.coronary[:, 1:2]
        label_tensor = torch.cat([self.cavity, coronary], dim=0)
        # use expand when possible to avoid a physical repeat copy (ddf has batch dim == 1)
        label_grid = ddf.expand(2, -1, -1, -1, -1)
        label_warped = self.warp_label(label_tensor, label_grid)
        
        cav_warped: Tensor = label_warped[0].unsqueeze(0)
        cor_warped: Tensor = label_warped[1].unsqueeze(0)

        return img_warped.half(), cav_warped.byte(), cor_warped.byte(), self._points_warp_and_to_world(ddf_inv, coronary_type)



class VolumeDVFReader(DataReader):
    def __init__(
        self,
        image_nii: Path,
        cavity_nii: Path,
        coronary_nii: Path,
        roi_json: Path,
        dvf_dir: Path,
        movement_enhancer: None | Type[MovementEnhancer] = None,
        recover_cropped_data: bool = True,
        precompute_ddf: bool = True,
    ):
        """Reads volume data and corresponding DVFs for 4D cardiac MRI, which usually generated from ..dvf module.
        
        the data is assumed to be in the following format:
        - image_nii: 3D volume image of one phase
        - cavity_nii: 3D cavity label image of one phase
        - coronary_nii: 3D coronary label image of one phase
        - roi_json: JSON file containing ROI information
        - dvf_dir: contains 3D DVF images for all phases
        | - phase_00.nii.gz   # do not use phase_0.nii which may cause ordering issues
        | - phase_01.nii.gz
        | - ...
        """
        if not torch.cuda.is_available() and torch.cuda.device_count() < 2:
            raise RuntimeError("No CUDA device available for VolumeDVFReader")
        
        self.device = torch.device("cuda:1")
        self.roi = ROI.from_json(roi_json)
        self.recover_cropped_data = recover_cropped_data
        self.precompute_ddf = precompute_ddf
        self.dvf2ddf = DVF2DDF(num_steps=5, mode="bilinear", padding_mode="zeros").to(self.device)
        
        self._load_3d_data(image_nii, cavity_nii, coronary_nii)
        self._load_dvf_and_init_warpper(dvf_dir, movement_enhancer)

    @staticmethod
    def _preprocess_image(image: Tensor) -> Tensor:
        # Some CTs may use -3023 or -2000 as 'sentinel' to mark invalid voxels
        # Set all invalid voxels to the minimum value, usually is air
        sentinel_mask = (image <= -2000)  
        min_value = image[~sentinel_mask].min()
        image[sentinel_mask] = min_value    
        
        # Add the offset of 1024 that is commonly used in CT
        if image.min() < -1000:
            image +=1024
        # These two values should be described in DICOM head
        # However, nifti does not have this field, we can only guess it from the value range
        
        return image
    
    def _load_3d_data(self, image_nii: Path, cavity_nii: Path, coronary_nii: Path):
        image, affine = load_nifti(image_nii)
        assert affine is not None
        image = self._preprocess_image(image)
        
        cavity, cavity_affine = load_nifti(cavity_nii, is_label=True)
        coronary, coronary_affine = load_nifti(coronary_nii, is_label=True)
        assert np.allclose(affine, cavity_affine)
        assert np.allclose(affine, coronary_affine)
        
        lca, rca = separate_coronary(coronary, self.device)        
        self.origin_data = _Data(
            self.device, image, cavity, affine, 
            lca, rca
        )

        self._origin_volume_size = image.shape[2:]   #type: ignore
        self._origin_volume_affine = affine
        
        # --- Crop Data to ROI to Match DVF ---
        def crop(x: Tensor) -> Tensor:
            return self.roi.crop_on_data(x.clone())

        self.cropped_data = _Data(
            self.device, crop(image), crop(cavity), self.roi.get_affine_after_crop(affine),
            crop(lca), crop(rca)
        )

    def _load_dvf_and_init_warpper(self, dvf_dir: Path, movement_enhancer: None | Type[MovementEnhancer] = None):
        # --- Load and Enhance DVFs ---
        if movement_enhancer is None:
            self.enhancer: Callable[[Tensor], Tensor] = lambda x: x
        else:
            self.enhancer = movement_enhancer(self.cropped_data.cavity, self.cropped_data.all_coronary_label, self.device)

        dvf_list: list[Tensor] = []
        ddf_list: list[Tensor] = []

        # prepare a dvf2ddf module on device for precomputation if requested
        for dvf_file in tqdm(sorted(dvf_dir.glob("*.nii.gz")), desc="loading and preprocessing DVFs"):
            # image saved as spacing of 1mm, so we need to zoom it back to the original spacing
            # save as half-precision to save memory
            dvf = load_and_zoom_dvf(dvf_file, self.roi, self.device).half()
            dvf = self.enhancer(dvf).float()
            
            if self.precompute_ddf:
                # compute DDF and inverse DDF on device once, store as CPU pinned to be used later
                with torch.no_grad():
                    ddf_batch: Tensor = self.dvf2ddf(torch.cat((dvf, -dvf), dim=0))
                ddf_list.append(ddf_batch.half().cpu().pin_memory())   # (2, 3, H, W, D)
            else:
                dvf_list.append(dvf.half().cpu().pin_memory())

            
        # TODO: for now only support fixed NUM_TOTAL_PHASE
        if self.precompute_ddf:
            self.n_phases = len(ddf_list)
        else:
            self.n_phases = len(dvf_list)
            
        assert self.n_phases == NUM_TOTAL_PHASE, f"For now, we only support {NUM_TOTAL_PHASE} phases. But the data has {self.n_phases} phases."
        
        # --- Initialize Warp Module ---
        if self.precompute_ddf:
            self.warpper = LazyCardiacDVFWarpModule(
                self.cropped_data, self.device,
                precomputed_ddf=True,
                ddf_list=ddf_list,
            ).to(self.device)
        else:
            self.warpper = LazyCardiacDVFWarpModule(
                self.cropped_data, self.device,
                precomputed_ddf=False,
                dvf_list=dvf_list
            ).to(self.device)
    
    
    def get_data(self, phase: CardiacPhase, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        coronary_type = CoronaryType(coronary_type)
        
        image, cavity, coronary_label, coronary_mesh = self.warpper(phase, coronary_type)
        if self.recover_cropped_data:
            image = self.roi.recover_cropped_tensor(image, background=self.origin_data.image)
            cavity = self.roi.recover_cropped_tensor(cavity)
            coronary_label = self.roi.recover_cropped_tensor(coronary_label)
            affine = self.origin_data.affine
            coronary_centering_affine = self.origin_data.lca_centering_affine if coronary_type == CoronaryType.LCA\
                else self.origin_data.rca_centering_affine
        else:
            affine = self.cropped_data.affine
            coronary_centering_affine = self.cropped_data.lca_centering_affine if coronary_type == CoronaryType.LCA\
                else self.cropped_data.rca_centering_affine
        
        return DataReaderResult(
            phase=phase,
            volume=image.cpu().to(torch.float32),
            cavity_label=cavity.cpu().to(torch.uint8),
            affine=affine,
            coronary = Coronary(
                type=coronary_type,
                label=coronary_label.cpu().to(torch.bool),
                centering_affine=coronary_centering_affine,
                mesh_in_world=coronary_mesh
            )
        )
    
    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        coronary_type = CoronaryType(coronary_type)
        
        if self.recover_cropped_data:
            data = self.origin_data
        else:
            data = self.cropped_data
        
        if coronary_type == CoronaryType.LCA:
            coronary_label = data.lca
            coronary_centering_affine = data.lca_centering_affine
        else:
            coronary_label = data.rca
            coronary_centering_affine = data.rca_centering_affine
        
        return DataReaderResult(
            phase=CardiacPhase(0),
            volume=data.image.cpu().to(torch.float32),
            cavity_label=data.cavity.cpu().to(torch.uint8),
            affine=data.affine,
            coronary=Coronary(
                type=coronary_type,
                label=coronary_label.cpu().to(torch.bool),
                centering_affine=coronary_centering_affine,
                mesh_in_world=get_mesh_in_world(coronary_label, coronary_centering_affine)
            )
        )
    
    def save_roi(self, output_dir: Path):
        import json
        json_data = self.roi.to_dict()
        with open(output_dir / "roi.json", "w") as f:
            json.dump(json_data, f)
    
    def save_all_warped_images(self, output_dir: Path):
        # TODO
        raise NotImplementedError()
    
    
    @property
    def lca_centering_affine(self) -> np.ndarray:
        if self.recover_cropped_data:
            return self.origin_data.lca_centering_affine
        else:
            return self.cropped_data.lca_centering_affine
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        if self.recover_cropped_data:
            return self.origin_data.rca_centering_affine
        else:
            return self.cropped_data.rca_centering_affine
    
    
    @property
    @override
    def volume_size(self) -> tuple[int, int, int]:
        if self.recover_cropped_data:
            return self._origin_volume_size
        else:
            return self.warpper.image.shape[-3:]    # type: ignore
    
    @property
    @override
    def volume_affine(self) -> np.ndarray:
        if self.recover_cropped_data:
            return self._origin_volume_affine
        else:
            return self.cropped_data.affine