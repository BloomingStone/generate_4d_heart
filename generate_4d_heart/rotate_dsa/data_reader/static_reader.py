from pathlib import Path
from typing import Literal

import torch
import numpy as np

from .data_reader import DataReader, DataReaderResult, Coronary, load_nifti
from generate_4d_heart.rotate_dsa.contrast_simulator import ContrastSimulator, IdentityContrast
from ..cardiac_phase import CardiacPhase
from ..types import CoronaryType

class StaticVolumeReader(DataReader):
    def __init__(
        self, 
        volume_path: Path,
        cavity_path: Path,
        coronary_path: Path,
        contrast_simulator: ContrastSimulator,
        device: torch.device = torch.device("cuda:0")
    ):
        self.volume_nii = volume_path
        self.cavity_nii = cavity_path
        self.coronary_nii = coronary_path
        if not torch.cuda.is_available() and torch.cuda.device_count() < 2:
            raise RuntimeError("No CUDA device available for VolumeDVFReader")
        self.device = device
        # default contrast simulator: identity (no change)
        self.contrast_simulator = contrast_simulator
        
        volume, affine = load_nifti(self.volume_nii)
        assert affine is not None
        
        cavity, cavity_affine = load_nifti(self.cavity_nii, is_label=True)
        coronary, coronary_affine = load_nifti(self.coronary_nii, is_label=True)
        assert np.allclose(affine, cavity_affine)
        assert np.allclose(affine, coronary_affine)

        self._data = self._Data.init_from_volume(
            volume=volume,
            cavity=cavity,
            coronary=coronary,
            affine=affine,
            running_device=self.device,
            contrast_simulator=self.contrast_simulator
        )

        self._origin_volume_size = volume.shape[2:]   #type: ignore
        self._origin_volume_affine = affine
        
        self.n_phases: int = 1

    
    def get_data(
        self,
        phase: CardiacPhase,
        coronary_type: CoronaryType | Literal["LCA", "RCA"],
        global_time: float = 0.0,
    ) -> DataReaderResult:
        cor_type = CoronaryType(coronary_type)
        cor = self._data[cor_type]
        volume = cor.volume 
        
        if self.contrast_simulator.contrast_change_over_time:
            # For dynamic contrast simulators, simulate with time information
            volume = self.contrast_simulator.simulate_with_time(
                volume, 
                self._data.cavity, 
                cor.label, 
                global_time
            )
        
        return DataReaderResult(
            phase=phase,
            cavity_label=self._data.cavity.cpu().to(torch.uint8),
            affine=self._origin_volume_affine,
            coronary=Coronary(
                type=cor_type,
                volume=volume.cpu(),
                label=cor.label.cpu().to(torch.bool),
                original_affine=cor.original_affine,
                centering_affine=cor.centering_affine,
                mesh_original=cor.mesh_original
            )
        )

    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        return self.get_data(CardiacPhase(0), coronary_type)

    @property
    def lca_centering_affine(self) -> np.ndarray:
        return self._data.lca.centering_affine
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        return self._data.rca.centering_affine


class StaticLabelReader(StaticVolumeReader):
    def __init__(
        self, 
        cavity_path: Path,
        coronary_path: Path,
        contrast_simulator: ContrastSimulator,
        device: torch.device = torch.device("cuda:0")
    ):
        self.cavity_nii = cavity_path
        self.coronary_nii = coronary_path
        if not torch.cuda.is_available() and torch.cuda.device_count() < 2:
            raise RuntimeError("No CUDA device available for VolumeDVFReader")
        self.device = device
        # default contrast simulator: identity (no change)
        self.contrast_simulator = contrast_simulator
        
        self.volume_nii = None  # StaticLabelReader does not use volume data

        
        cavity, cavity_affine = load_nifti(self.cavity_nii, is_label=True)
        coronary, coronary_affine = load_nifti(self.coronary_nii, is_label=True)
        assert np.allclose(coronary_affine, cavity_affine)

        self._data = self._Data.init_from_volume(
            volume=torch.zeros_like(cavity, dtype=torch.float32),  # StaticLabelReader does not use volume data, but we need to provide a dummy volume for contrast simulation
            cavity=cavity,
            coronary=coronary,
            affine=cavity_affine,
            running_device=self.device,
            contrast_simulator=None
        )
        if not self.contrast_simulator.contrast_change_over_time:
            print("Applying STATIC contrast simulation to original volume for StaticLabelReader...")
            self._data.lca.volume = self.contrast_simulator.simulate(
                self._data.lca.volume, 
                self._data.cavity, 
                self._data.lca.label
            )
            self._data.rca.volume = self.contrast_simulator.simulate(
                self._data.rca.volume, 
                self._data.cavity, 
                self._data.rca.label
            )

        self._origin_volume_size = cavity.shape[2:]   #type: ignore
        self._origin_volume_affine = cavity_affine
        
        self.n_phases: int = 1
        