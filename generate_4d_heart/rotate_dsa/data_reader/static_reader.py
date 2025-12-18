from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field

import torch
from torch import Tensor
import numpy as np

from .data_reader import DataReader, DataReaderResult, Coronary, separate_coronary, load_nifti, get_coronary_centering_affine, get_mesh_in_world
from ... import NUM_TOTAL_PHASE, NUM_TOTAL_CAVITY_LABEL
from ..cardiac_phase import CardiacPhase
from ..types import CoronaryType


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


class StaticVolumeReader(DataReader):
    def __init__(
        self, 
        image_path: Path,
        cavity_path: Path,
        coronary_path: Path,
        device: torch.device = torch.device("cuda:0")
    ):
        self.image_nii = image_path
        self.cavity_nii = cavity_path
        self.coronary_nii = coronary_path
        if not torch.cuda.is_available() and torch.cuda.device_count() < 2:
            raise RuntimeError("No CUDA device available for VolumeDVFReader")
        self.device = device
        self._load_3d_data()
        
        self.n_phases: int = 1
        
        self.meshes = {
            CoronaryType.LCA: get_mesh_in_world(self.origin_data.lca, self.origin_data.lca_centering_affine, self.device),
            CoronaryType.RCA: get_mesh_in_world(self.origin_data.rca, self.origin_data.rca_centering_affine, self.device)
        }
    
    def _load_3d_data(self):
        image, affine = load_nifti(self.image_nii)
        assert affine is not None
        
        cavity, cavity_affine = load_nifti(self.cavity_nii, is_label=True)
        coronary, coronary_affine = load_nifti(self.coronary_nii, is_label=True)
        assert np.allclose(affine, cavity_affine)
        assert np.allclose(affine, coronary_affine)
        
        lca, rca = separate_coronary(coronary, self.device)        
        self.origin_data = _Data(
            self.device, image, cavity, affine, 
            lca, rca
        )

        self._origin_volume_size = image.shape[2:]   #type: ignore
        self._origin_volume_affine = affine

    
    def get_data(self, phase: CardiacPhase, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        coronary_type = CoronaryType(coronary_type)
            
        if coronary_type == CoronaryType.LCA:
            coronary_label = self.origin_data.lca
            coronary_centering_affine = self.origin_data.lca_centering_affine
        else:
            coronary_label = self.origin_data.rca
            coronary_centering_affine = self.origin_data.lca_centering_affine
        
        return DataReaderResult(
            phase=phase,
            volume=self.origin_data.image.cpu(),
            cavity_label=self.origin_data.cavity.cpu().to(torch.uint8),
            affine=self._origin_volume_affine,
            coronary=Coronary(
                type=coronary_type,
                label=coronary_label.cpu().to(torch.bool),
                centering_affine=coronary_centering_affine,
                mesh_in_world=self.meshes[coronary_type]
            )
        )

    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        return self.get_data(CardiacPhase(0), coronary_type)

    @property
    def lca_centering_affine(self) -> np.ndarray:
        return self.origin_data.lca_centering_affine
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        return self.origin_data.rca_centering_affine

class StaticLabelReader(DataReader):
    def __init__(
        self, 
        cavity_path: Path,
        coronary_path: Path,
        device: torch.device = torch.device("cuda:0")
    ):
        self.device = device
        self.n_phases: int = NUM_TOTAL_PHASE
        self.cavity, _ = load_nifti(cavity_path, is_label=True)
        coronary, self._origin_volume_affine = load_nifti(coronary_path, is_label=True)
        self.lca_label, self.rca_label = separate_coronary(coronary, self.device)
        self._origin_volume_size = self.cavity.shape[-3:]   #type: ignore
        self._lca_centering_affine = get_coronary_centering_affine(self.lca_label, self._origin_volume_affine, self.device)
        self._rca_centering_affine = get_coronary_centering_affine(self.rca_label, self._origin_volume_affine, self.device)
        self.meshes = {
            CoronaryType.LCA: get_mesh_in_world(self.lca_label, self._lca_centering_affine, self.device),
            CoronaryType.RCA: get_mesh_in_world(self.rca_label, self._rca_centering_affine, self.device)
        }
    
    
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
                mesh_in_world=self.meshes[coronary_type]
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