from pathlib import Path
from typing import Type

import torch
from monai.networks.blocks.warp import DVF2DDF, Warp
from tqdm import tqdm

from ... import NUM_TOTAL_PHASE
from ..movement_enhancer import MovementEnhancer, IdentityMovementEnhancer
from ...roi import ROI
from ..cardiac_phase import CardiacPhase
from .data_reader import (
    DataReader, DataReaderResult, separate_coronary, 
    load_nifti, load_nifti_with_roi_crop, load_and_zoom_dvf
)

import torch
import torch.nn as nn
from monai.networks.blocks.warp import DVF2DDF, Warp

import torch
import torch.nn as nn
from monai.networks.blocks.warp import DVF2DDF, Warp


class LazyCardiacDVFWarpModule(nn.Module):
    """
    Compute warped image/labels from DVFs stored on CPU.
    Uses async transfer and small GPU cache for efficiency.
    """

    def __init__(
        self, 
        dvf_list: list[torch.Tensor],  # list of [3, D, H, W], stored in CPU (pinned)
        image: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        device: torch.device
    ):
        super().__init__()

        self.dvf_list_cpu = [dvf.pin_memory() for dvf in dvf_list]
        self.device = device

        # keep only 2 recent DVFs on GPU to save memory
        self.gpu_cache = {}

        self.image = image.half().to(device, non_blocking=True)
        self.cavity = cavity_label.half().to(device, non_blocking=True)
        self.coronary = coronary_label.half().to(device, non_blocking=True)

        self.dvf2ddf = DVF2DDF(num_steps=5, mode="bilinear", padding_mode="zeros").to(device).half()
        self.warp_image = Warp(mode="bilinear", padding_mode="zeros").to(device).half()
        self.warp_label = Warp(mode="nearest", padding_mode="zeros").to(device).half()

        self.n_phases = len(dvf_list)

    def _get_dvf(self, idx: int) -> torch.Tensor:
        """
        Load DVF to GPU if not cached. Uses async copy.
        """
        if idx in self.gpu_cache:
            return self.gpu_cache[idx]

        dvf_cpu = self.dvf_list_cpu[idx]
        dvf_gpu = dvf_cpu.to(self.device, non_blocking=True)

        # maintain small LRU-like cache
        if len(self.gpu_cache) >= 2:
            dvf_old = self.gpu_cache.pop(next(iter(self.gpu_cache)))  # remove oldest
            dvf_old.cpu()
        
        self.gpu_cache[idx] = dvf_gpu
        return dvf_gpu

    @torch.inference_mode()
    def forward(self, phase: CardiacPhase, prefetch_next_phase: CardiacPhase|None = None):
        """
        Compute warped tensors for the given phase ∈ [0,1)
        """
        idx0 = phase.closest_index_floor(self.n_phases)
        idx1 = phase.closest_index_ceil(self.n_phases)
        w = float(phase) * self.n_phases - idx0

        # --- async copy from CPU ---
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            dvf0 = self._get_dvf(idx0)
            dvf1 = self._get_dvf(idx1)
        torch.cuda.current_stream().wait_stream(stream)

        dvf = dvf0 * (1 - w) + dvf1 * w
        ddf: torch.Tensor = self.dvf2ddf(dvf)

        img_warped: torch.Tensor = self.warp_image(self.image, ddf)
        label_tensor = torch.cat([self.cavity, self.coronary], dim=0)
        label_warped = self.warp_label(label_tensor, ddf.repeat(2,1,1,1,1))
        
        cav_warped: torch.Tensor = label_warped[0].unsqueeze(0)
        cor_warped: torch.Tensor = label_warped[1].unsqueeze(0)

        return img_warped.half(), cav_warped.byte(), cor_warped.byte(), ddf.half()


class VolumeDVFReader(DataReader):
    def __init__(
        self,
        image_nii: Path,
        cavity_nii: Path,
        coronary_nii: Path,
        roi_json: Path,
        dvf_dir: Path,
        movement_enhancer: Type[MovementEnhancer] = IdentityMovementEnhancer
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.roi = ROI.from_json(roi_json)
        
        image, self.origin_image_affine = load_nifti_with_roi_crop(image_nii, self.roi, is_label=False)
        cavity_label, _ = load_nifti_with_roi_crop(cavity_nii, self.roi, is_label=True)
        coronary_label, _ = load_nifti_with_roi_crop(coronary_nii, self.roi, is_label=True)

        self.image_before_cropped, _ = load_nifti(image_nii, is_label=False) # (1,1,D,H,W)
        self.origin_image_size = self.image_before_cropped.shape[2:]   #type: ignore
        
        self.movement_enhancer = movement_enhancer
        self.enhancer = self.movement_enhancer(cavity_label, coronary_label)

        # image saved as spacing of 1mm, so we need to zoom it back to the original spacing
        dvf_list: list[torch.Tensor] = []
        # TODO: for now only support fixed NUM_TOTAL_PHASE
        for dvf_file in tqdm(sorted(dvf_dir.glob("*.nii.gz")), desc="loading and updating DVFs"):
            dvf_tensor = load_and_zoom_dvf(dvf_file, self.roi, device).half()
            dvf_tensor = self.enhancer.enhance(dvf_tensor)
            dvf_list.append(dvf_tensor)
        
        self.n_phases = len(dvf_list)
        assert self.n_phases == NUM_TOTAL_PHASE, f"For now, we only support {NUM_TOTAL_PHASE} phases. But the data has {self.n_phases} phases."
        
        self.warpper = LazyCardiacDVFWarpModule(
            dvf_list, image, cavity_label, coronary_label, device
        ).to(device)
    
    def get_data(self, phase: CardiacPhase) -> DataReaderResult:
        image, cavity, coronary, _ = self.warpper(phase)
        image = self.roi.recover_cropped_tensor(image, self.image_before_cropped)
        cavity = self.roi.recover_cropped_tensor(cavity)
        coronary = self.roi.recover_cropped_tensor(coronary)
        lca, rca = separate_coronary(coronary)

        return DataReaderResult(
            phase=phase,
            volume=image.cpu(),
            cavity_label=cavity.cpu(),
            lca_label=lca.cpu(),
            rca_label=rca.cpu(),
            affine=self.origin_image_affine
        )
    
    def save_roi(self, output_dir: Path):
        import json
        json_data = self.roi.to_dict()
        with open(output_dir / "roi.json", "w") as f:
            json.dump(json_data, f)
    
    def save_all_warped_images(self, output_dir: Path):
        # TODO
        raise NotImplementedError()