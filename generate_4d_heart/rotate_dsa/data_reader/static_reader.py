from pathlib import Path
from typing import Literal

import torch

from .data_reader import DataReader, DataReaderResult, separate_coronary, load_nifti, get_coronary_centering_affine
from ... import NUM_TOTAL_PHASE, NUM_TOTAL_CAVITY_LABEL
from ..cardiac_phase import CardiacPhase

class StaticVolumeReader(DataReader):
    def __init__(
        self, 
        image_path: Path,
        cavity_path: Path,
        coronary_path: Path
    ):
        self.n_phases: int = NUM_TOTAL_PHASE
        self.volume, self._origin_volume_affine = load_nifti(image_path)
        self.cavity, _ = load_nifti(cavity_path, is_label=True)
        coronary, _ = load_nifti(coronary_path, is_label=True)
        self.lca_label, self.rca_label = separate_coronary(coronary)
        self._origin_volume_size = self.volume.shape[-3:]   #type: ignore
        self.lca_centering_affine = get_coronary_centering_affine(self.lca_label, self._origin_volume_affine)
        self.rca_centering_affine = get_coronary_centering_affine(self.rca_label, self._origin_volume_affine)
    
    def get_data(self, phase: CardiacPhase) -> DataReaderResult:
        return DataReaderResult(
            phase=phase,
            volume=self.volume.cpu(),
            cavity_label=self.cavity.cpu().to(torch.uint8),
            lca_label=self.lca_label.cpu().to(torch.bool),
            rca_label=self.rca_label.cpu().to(torch.bool),
            affine=self._origin_volume_affine,
            lca_centering_affine=self.lca_centering_affine,
            rca_centering_affine=self.rca_centering_affine
        )
    
    def get_phase_0_data(self) -> DataReaderResult:
        return self.get_data(CardiacPhase(0))

class StaticLabelReader(DataReader):
    def __init__(
        self, 
        cavity_path: Path,
        coronary_path: Path,
        coronary_type: Literal["LCA", "RCA"],
    ):
        self.n_phases: int = NUM_TOTAL_PHASE
        self.cavity, self._origin_volume_affine = load_nifti(cavity_path, is_label=True)
        coronary, _ = load_nifti(coronary_path, is_label=True)
        self.lca_label, self.rca_label = separate_coronary(coronary)
        self._origin_volume_size = self.cavity.shape[-3:]   #type: ignore
        self.volume = self.lca_label if coronary_type == "LCA" else self.rca_label
        self.lca_centering_affine = get_coronary_centering_affine(self.lca_label, self._origin_volume_affine)
        self.rca_centering_affine = get_coronary_centering_affine(self.rca_label, self._origin_volume_affine)
    
    def get_data(self, phase: CardiacPhase) -> DataReaderResult:
        return DataReaderResult(
            phase=phase,
            volume=self.volume.cpu().to(torch.float),
            cavity_label=self.cavity.cpu().to(torch.uint8),
            lca_label=self.lca_label.cpu().to(torch.bool),
            rca_label=self.rca_label.cpu().to(torch.bool),
            affine=self._origin_volume_affine,
            lca_centering_affine=self.lca_centering_affine,
            rca_centering_affine=self.rca_centering_affine
        )
    
    def get_phase_0_data(self) -> DataReaderResult:
        return self.get_data(CardiacPhase(0))