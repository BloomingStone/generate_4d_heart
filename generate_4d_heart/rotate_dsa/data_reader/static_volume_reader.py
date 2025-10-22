from pathlib import Path

import torch

from .data_reader import DataReader, DataReaderResult, separate_coronary, load_nifti
from ... import NUM_TOTAL_PHASE, NUM_TOTAL_CAVITY_LABEL
from ..cardiac_phase import CardiacPhase

class StaticVolumeReader(DataReader):
    def __init__(
        self, 
        image_path: Path,
        cavity_path: Path,
        coronary_path: Path,
    ):
        self.n_phases: int = NUM_TOTAL_PHASE
        self.volume, self.origin_image_affine = load_nifti(image_path)
        self.cavity, _ = load_nifti(cavity_path, is_label=True)
        coronary, _ = load_nifti(coronary_path, is_label=True)
        self.lca_label, self.rca_label = separate_coronary(coronary)
        self.origin_image_size = self.volume.shape[-3:]   #type: ignore
    
    def get_data(self, phase: CardiacPhase) -> DataReaderResult:
        return DataReaderResult(
            phase=phase,
            volume=self.volume.cpu(),
            cavity_label=self.cavity.cpu().to(torch.uint8),
            lca_label=self.lca_label.cpu().to(torch.bool),
            rca_label=self.rca_label.cpu().to(torch.bool),
            affine=self.origin_image_affine
        )