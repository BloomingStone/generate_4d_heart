from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field
from multiprocessing import get_context
import os

import torch
from torch import Tensor
import numpy as np
import pyvista as pv
import torchcpd
from tqdm import tqdm

from .data_reader import DataReader, DataReaderResult, Coronary, separate_coronary, load_nifti, get_coronary_centering_affine, get_mesh_in_world
from generate_4d_heart.rotate_dsa.contrast_simulator import ContrastSimulator
from ... import NUM_TOTAL_PHASE
from ..cardiac_phase import CardiacPhase
from ..types import CoronaryType

class VolumesReader(DataReader):
    def __init__(
        self, 
        image_dir: Path,
        cavity_dir: Path,
        coronary_dir: Path,
        contrast_simulator: ContrastSimulator,
        device: torch.device = torch.device("cuda:0")
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
        self.device = device
        # default contrast simulator: identity
        self.contrast_simulator = contrast_simulator
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
        
        self.data: list[DataReader._Data] = []
        for img_file, cav_file, cor_file in tqdm(
            zip(image_file_list, cavity_file_list, coronary_file_list),
            desc="Loading data", total=len(image_file_list)
        ):
            volume, affine = load_nifti(img_file)
            if volume.max() < 2.0:
                print("Assuming image is normalized to [0,1] (from XCAT), recovering original intensity to CTA by multiplying 65536 and subtracting 1000")
                volume *= 2**16  # recover image uses float to represent 16 bit
                volume -= 1000
            
            cavity_label, _ = load_nifti(cav_file, is_label=True)
            coronary_label, _ = load_nifti(cor_file, is_label=True)

            self.data.append(DataReader._Data.init_from_volume(
                volume=volume,
                cavity=cavity_label,
                coronary=coronary_label,
                affine=affine,
                running_device=self.device,
                contrast_simulator=self.contrast_simulator
            ))
        
        # Align coronary centering affine across phases
        self.lca_centering_affine_0 = self.data[0].coronary[CoronaryType.LCA].centering_affine
        self.rca_centering_affine_0 = self.data[0].coronary[CoronaryType.RCA].centering_affine
        for data_at_phase in self.data:
            data_at_phase.coronary[CoronaryType.LCA].centering_affine = self.lca_centering_affine_0
            data_at_phase.coronary[CoronaryType.RCA].centering_affine = self.rca_centering_affine_0


    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        return self._get_data_at_index(0, CardiacPhase(0), CoronaryType(coronary_type))

    def _get_data_at_index(self, index: int, phase: CardiacPhase, coronary_type: CoronaryType, global_time: float|None=None) -> DataReaderResult:
        data = self.data[index]
        cor = data[coronary_type]
        
        volume = cor.volume
        if self.contrast_simulator.contrast_change_over_time and global_time is not None:
            volume = self.contrast_simulator.simulate_with_time(
                float(global_time),
                volume,
                data.cavity,
                cor.label,
                cor.centering_affine,
            )

        return DataReaderResult(
            phase=phase,
            cavity_label=data.cavity.cpu().to(torch.uint8),
            affine=self._origin_volume_affine,
            coronary=Coronary(
                type=coronary_type,
                volume=volume.cpu(),
                label=cor.label.cpu().to(torch.bool),
                original_affine=self._origin_volume_affine,
                centering_affine=cor.centering_affine,
                mesh_original=cor.mesh_original
            )
        )

    def get_data(
        self,
        phase: CardiacPhase,
        coronary_type: CoronaryType | Literal["LCA", "RCA"],
        global_time: float = 0.0,
    ) -> DataReaderResult:
        """
        Returns interpolated data for the given phase in [0,1).
        If phase matches an existing frame index exactly, returns cached result.
        Otherwise performs fast linear interpolation between the two nearest frames.
        """
        coronary_type = CoronaryType(coronary_type)
        
        idx0 = phase.lower_index(self.n_phases)
        idx1 = phase.upper_index(self.n_phases)
        
        w = float(phase)*self.n_phases - idx0  # interpolation weight [0,1)

        if w < 1e-6:  # exactly at frame idx0
            return self._get_data_at_index(idx0, phase, coronary_type, global_time)

        d0 = self.data[idx0]
        d1 = self.data[idx1]
        cor0 = d0[coronary_type]
        cor1 = d1[coronary_type]

        # === intensity volume interpolation ===
        vol_interp = (1 - w) * cor0.volume + w * cor1.volume

        # cavity has multiple labels, mesh's points is non-structured, can not be simply interpolated.  
        cavity = d0.cavity if w < 0.5 else d1.cavity
        mesh = cor0.mesh_original.copy() if w < 0.5 else cor1.mesh_original.copy()

        # === coronary label interpolation (LCA & RCA) ===
        coronary_label = ((1 - w) * cor0.label.float() + w * cor1.label.float()) > 0.5
        
        # If simulator is dynamic, apply simulate_with_time to coronary region now
        if self.contrast_simulator.contrast_change_over_time:
            vol_interp = self.contrast_simulator.simulate_with_time(
                float(global_time),
                vol_interp,
                cavity,
                coronary_label,
                cor0.centering_affine,  # already aligned in __init__
            )

        return DataReaderResult(
            phase=phase,
            cavity_label=cavity.cpu().to(torch.uint8),
            affine=self._origin_volume_affine,
            coronary=Coronary(
                type=coronary_type,
                volume=vol_interp.cpu(),
                label=coronary_label.cpu().to(torch.bool),
                original_affine=self._origin_volume_affine,
                centering_affine=cor0.centering_affine,  # already aligned in __init__
                mesh_original=mesh
            )
        )
    
    @property
    def lca_centering_affine(self) -> np.ndarray:
        return self.lca_centering_affine_0
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        return self.rca_centering_affine_0

