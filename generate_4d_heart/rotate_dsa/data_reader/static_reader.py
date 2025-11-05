from pathlib import Path
from typing import Literal

import torch
import numpy as np

from .data_reader import DataReader, DataReaderResult, Coronary, separate_coronary, load_nifti, get_coronary_centering_affine, get_mesh_in_world
from ... import NUM_TOTAL_PHASE, NUM_TOTAL_CAVITY_LABEL
from ..cardiac_phase import CardiacPhase
from ..types import CoronaryType

class StaticVolumeReader(DataReader):
    def __init__(
        self, 
        image_path: Path,
        cavity_path: Path,
        coronary_path: Path
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
        else:
            raise RuntimeError("No CUDA device available for StaticVolumeReader")
        self.n_phases: int = NUM_TOTAL_PHASE
        self.volume, self._origin_volume_affine = load_nifti(image_path)
        self.cavity, _ = load_nifti(cavity_path, is_label=True)
        coronary, _ = load_nifti(coronary_path, is_label=True)
        self.lca_label, self.rca_label = separate_coronary(coronary, self.device)
        self._origin_volume_size = self.volume.shape[-3:]   #type: ignore
        self._lca_centering_affine = get_coronary_centering_affine(self.lca_label, self._origin_volume_affine, self.device)
        self._rca_centering_affine = get_coronary_centering_affine(self.rca_label, self._origin_volume_affine, self.device)
    
    def get_data(self, phase: CardiacPhase, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        coronary_type = CoronaryType(coronary_type)
            
        if coronary_type == CoronaryType.LCA:
            coronary_label = self.lca_label
            coronary_centering_affine = self._lca_centering_affine
        else:
            coronary_label = self.rca_label
            coronary_centering_affine = self._rca_centering_affine
        
        return DataReaderResult(
            phase=phase,
            volume=self.volume.cpu(),
            cavity_label=self.cavity.cpu().to(torch.uint8),
            affine=self._origin_volume_affine,
            coronary=Coronary(
                type=coronary_type,
                label=coronary_label.cpu().to(torch.bool),
                centering_affine=coronary_centering_affine,
                mesh_in_world=get_mesh_in_world(
                    coronary_label, 
                    coronary_centering_affine)
            )
        )

    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        return self.get_data(CardiacPhase(0), coronary_type)

    @property
    def lca_centering_affine(self) -> np.ndarray:
        return self._lca_centering_affine
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        return self._rca_centering_affine

class StaticLabelReader(DataReader):
    def __init__(
        self, 
        cavity_path: Path,
        coronary_path: Path,
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
        else:
            raise RuntimeError("No CUDA device available for StaticLabelReader")
        self.n_phases: int = NUM_TOTAL_PHASE
        self.cavity, _ = load_nifti(cavity_path, is_label=True)
        coronary, self._origin_volume_affine = load_nifti(coronary_path, is_label=True)
        self.lca_label, self.rca_label = separate_coronary(coronary, self.device)
        self._origin_volume_size = self.cavity.shape[-3:]   #type: ignore
        self._lca_centering_affine = get_coronary_centering_affine(self.lca_label, self._origin_volume_affine, self.device)
        self._rca_centering_affine = get_coronary_centering_affine(self.rca_label, self._origin_volume_affine, self.device)
    
    def get_data(self, phase: CardiacPhase, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        coronary_type = CoronaryType(coronary_type)
        
        if coronary_type == CoronaryType.LCA:
            coronary_label = self.lca_label
            coronary_centering_affine = self._lca_centering_affine
        else:
            coronary_label = self.rca_label
            coronary_centering_affine = self._rca_centering_affine
        
        return DataReaderResult(
            phase=phase,
            volume=coronary_label.cpu(),
            cavity_label=self.cavity.cpu().to(torch.uint8),
            affine=self._origin_volume_affine,
            coronary=Coronary(
                type=coronary_type,
                label=coronary_label.cpu().to(torch.bool),
                centering_affine=coronary_centering_affine,
                mesh_in_world=get_mesh_in_world(
                    coronary_label, 
                    coronary_centering_affine)
            )
        )

    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        return self.get_data(CardiacPhase(0), coronary_type)

    @property
    def lca_centering_affine(self) -> np.ndarray:
        return self._lca_centering_affine
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        return self._rca_centering_affine