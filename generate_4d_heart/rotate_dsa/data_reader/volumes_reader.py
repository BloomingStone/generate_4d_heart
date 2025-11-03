from pathlib import Path
from typing import Literal
from dataclasses import dataclass

import torch
from torch import Tensor
import numpy as np

from .data_reader import DataReader, DataReaderResult, Coronary, separate_coronary, load_nifti, get_coronary_centering_affine, get_mesh_in_world
from ... import NUM_TOTAL_PHASE, NUM_TOTAL_CAVITY_LABEL
from ..cardiac_phase import CardiacPhase
from ..types import CoronaryType

@dataclass
class _Data:
    phase: CardiacPhase
    volume: Tensor
    cavity_label: Tensor
    affine: np.ndarray
    lca_label: Tensor
    rca_label: Tensor

class VolumesReader(DataReader):
    def __init__(
        self, 
        image_dir: Path,
        cavity_dir: Path,
        coronary_dir: Path,
    ):
        """
        Volume data reader for 4D cardiac MRI.

        The data is assumed to be in the following format:
        - image_dir: contains 4D volume images
        | - phase_000.nii.gz   # do not use phase_0.nii which may cause ordering issues
        | - phase_001.nii.gz
        | - ...
        - cavity_dir: contains 4D cavity label images
        | - phase_000.nii.gz
        | - phase_001.nii.gz
        | - ...
        - coronary_dir: contains 4D coronary label images
        | - phase_000.nii.gz
        | - phase_001.nii.gz
        | - ...
        """
        image_file_list = sorted(image_dir.glob("*.nii*"))
        cavity_file_list = sorted(cavity_dir.glob("*.nii*"))
        coronary_file_list = sorted(coronary_dir.glob("*.nii*"))
        
        assert len(image_file_list) == len(cavity_file_list) == len(coronary_file_list), "Mismatch in number of files"
        
        self.n_phases = len(image_file_list)
        
        assert self.n_phases == NUM_TOTAL_PHASE, f"For now, we only support {NUM_TOTAL_PHASE} phases. But the data has {self.n_phases} phases."
        
        volume_0, affine_0 = load_nifti(image_file_list[0])
        self._origin_volume_size = volume_0.shape[2:]   #type: ignore
        self._origin_volume_affine = affine_0
        assert len(self._origin_volume_size) == 3, f"Image size must be 3D, but got {self._origin_volume_size}"
        
        self.data: list[_Data] = []
        for index, (img_file, cav_file, cor_file) in enumerate(zip(image_file_list, cavity_file_list, coronary_file_list)):
            phase = CardiacPhase.from_index(index, self.n_phases)

            volume, affine = load_nifti(img_file)
            if volume.max() < 2.0:
                volume *= 2**16  # recover image uses float to represent 16 bit
            
            sentinel_mask = (volume <= -2000)  # Some CTs may use -3023 or -2000 as 'sentinel' to mark invalid voxels
            min_value = volume[~sentinel_mask].min()
            volume[sentinel_mask] = min_value
            
            if volume.min() < -1000:
                volume += 1024   # add the offset of 1024

            cavity_label, _ = load_nifti(cav_file, is_label=True)
            coronary_label, _ = load_nifti(cor_file, is_label=True)

            lca_label, rca_label = separate_coronary(coronary_label)
            
            if index == 0:
                self._lca_centering_affine = get_coronary_centering_affine(lca_label, self._origin_volume_affine)
                self._rca_centering_affine = get_coronary_centering_affine(rca_label, self._origin_volume_affine)

            self.data.append(_Data(
                phase=phase,
                volume=volume.cpu(),
                cavity_label=cavity_label.cpu(),
                lca_label=lca_label.cpu(),
                rca_label=rca_label.cpu(),
                affine=affine,
            ))
    

    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        if isinstance(coronary_type, str):
            coronary_type = CoronaryType(coronary_type)
        return self._get_data_at_index(0, coronary_type)

    def _get_data_at_index(self, index: int, coronary_type: CoronaryType) -> DataReaderResult:
        data = self.data[index]
        if coronary_type == CoronaryType.LCA:
            coronary_label = data.lca_label
            coronary_centering_affine = self._lca_centering_affine
        else:
            coronary_label = data.rca_label
            coronary_centering_affine = self._rca_centering_affine
        
        return DataReaderResult(
            phase=data.phase,
            volume=coronary_label.cpu(),
            cavity_label=data.cavity_label.cpu().to(torch.uint8),
            affine=self._origin_volume_affine,
            coronary=Coronary(
                type=coronary_type,
                label=coronary_label.cpu().to(torch.bool),
                centering_affine=coronary_centering_affine,
                mesh_in_world=get_mesh_in_world(
                    coronary_label, 
                    self._origin_volume_affine)
            )
        )

    def get_data(self, phase: CardiacPhase, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        """
        Returns interpolated data for the given phase in [0,1).
        If phase matches an existing frame index exactly, returns cached result.
        Otherwise performs fast linear interpolation between the two nearest frames.
        """
        if isinstance(coronary_type, str):
            coronary_type = CoronaryType(coronary_type)
        
        idx0 = phase.closest_index_floor(self.n_phases)
        idx1 = phase.closest_index_ceil(self.n_phases)
        
        w = float(phase)*self.n_phases - idx0  # interpolation weight [0,1)

        if w < 1e-6:  # exactly at frame idx0
            return self._get_data_at_index(idx0, coronary_type)

        d0 = self.data[idx0]
        d1 = self.data[idx1]

        # === intensity volume interpolation ===
        vol_interp = (1 - w) * d0.volume + w * d1.volume

        # === cavity label interpolation ===
        # use soft interpolation then rounding
        if w < 0.5:
            cav_interp = d0.cavity_label
        else:
            cav_interp = d1.cavity_label
        cav_interp = torch.clamp(cav_interp, 0, NUM_TOTAL_CAVITY_LABEL)

        # === coronary label interpolation (LCA & RCA) ===
        # binary -> linear then threshold (faster than morphological blending)
        lca_interp = ((1 - w) * d0.lca_label.float() + w * d1.lca_label.float()) > 0.5
        rca_interp = ((1 - w) * d0.rca_label.float() + w * d1.rca_label.float()) > 0.5

        if coronary_type == CoronaryType.LCA:
            coronary_label = lca_interp
            coronary_centering_affine = self._lca_centering_affine
        else:
            coronary_label = rca_interp
            coronary_centering_affine = self._rca_centering_affine
        
        return DataReaderResult(
            phase=phase,
            volume=vol_interp.cpu(),
            cavity_label=cav_interp.cpu().to(torch.uint8),
            affine=self._origin_volume_affine,
            coronary=Coronary(
                type=coronary_type,
                label=coronary_label.cpu().to(torch.bool),
                centering_affine=coronary_centering_affine,
                mesh_in_world=get_mesh_in_world(
                    coronary_label, 
                    self._origin_volume_affine)
            )
        )
    
    @property
    def lca_centering_affine(self) -> np.ndarray:
        return self._lca_centering_affine
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        return self._rca_centering_affine

