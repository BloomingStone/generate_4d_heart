from typing import Literal, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from pprint import pprint

import numpy as np
import torch

from generate_4d_heart.med_image import MedImage, Centering
from generate_4d_heart.rotate_dsa.data_reader.data_reader import (
    DataReader, DataReaderResult, Coronary, CenteringMode,
    load_nifti
)
from generate_4d_heart.ssm import SSMReader
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.types import CoronaryType
from generate_4d_heart.rotate_dsa.contrast_simulator import ContrastSimulator, IdentityContrast
from generate_4d_heart.rotate_dsa.data_reader.rbf_reader import KDTreeRBF


logger = logging.getLogger(__name__)

@dataclass
class RBFLabelReader(DataReader):
    """
    Simplified RBFReader that only reads label data (cavity and coronary) and generates deformed coronary mesh using RBF, without reading volume data. This is useful for generating simplified DRR/DSA from label data.
    """
    
    cavity_nii: Path
    coronary_nii: Path
    
    ssm_reader: SSMReader= field(default_factory=SSMReader)
    
    device: torch.device = field(default_factory= lambda: torch.device("cuda:0"))

    attenuation_transition: float = 5.0
    
    centering_mode: CenteringMode = "coronary_center"
    
    transform_to_centering: np.ndarray | None = None
    
    contrast_simulator: ContrastSimulator = field(default_factory=IdentityContrast, init=False)

    def __post_init__(self):
        print(f"\nInitializing RBFReader with \n\t- cavity_nii={self.cavity_nii} \n\t- coronary_nii={self.coronary_nii}")
        print(f"\t- Centering mode: {self.centering_mode}")
        print(f"\t- Using contrast_simulator: {self.contrast_simulator}")
        if self.artifact_manager is not None:
            print(f"\t- Using artifact_manager:\n\t", end="")
            pprint(self.artifact_manager.config)
        
        logger.info(f"Loading data from {self.cavity_nii}, {self.coronary_nii}")
        self.n_phases = self.ssm_reader.n_phases
        
        print("Loading NIfTI files...")
        cavity, cavity_affine = load_nifti(self.cavity_nii, is_label=True)
        affine = cavity_affine
        cavity_np = cavity.squeeze().cpu().numpy()
        coronary, coronary_affine = load_nifti(self.coronary_nii, is_label=True)
        assert np.allclose(affine, cavity_affine)
        assert np.allclose(affine, coronary_affine)        
        volume = torch.zeros_like(cavity, dtype=torch.float32)  # placeholder, as volume is not used in label reader
        self._origin_volume_size = volume.shape[2:]   #type: ignore
        self._origin_volume_affine = affine
        
        print("Initializing origin data...")
        self.origin_data = self._Data.init_from_volume(
            volume,cavity, coronary, affine, self
        )

        print("Loading SSM result and initializing RBF...")
        ssm_result = self.ssm_reader.load(
            cavity=cavity_np,
            affine=cavity_affine
        )

        # Extract coronary centre-lines for length-preserving RBF constraints
        print("Extracting coronary centre-lines for RBF length constraint...")
        centerline_lca = self.origin_data.lca.get_coronary_central_line("world")
        centerline_rca = self.origin_data.rca.get_coronary_central_line("world")
        self.rbf = KDTreeRBF(
            ssm_result.landmark,
            centerlines=[centerline_lca, centerline_rca],
            centerline_downsample=3,
            centerline_constraint_enabled=False,
            device=self.device,
        ).to(self.device).eval()
        
        print("RBF Label Reader initialized successfully.")

    @torch.no_grad()
    def get_static_data(
        self,
        coronary_type: CoronaryType | Literal["LCA", "RCA"],
    ) -> list[MedImage[Centering]]:
        return []

    @torch.no_grad()
    def get_dynamic_data(
        self,
        phase: CardiacPhase,
        coronary_type: CoronaryType | Literal["LCA", "RCA"],
        global_time: float = 0.0,
    ) -> DataReaderResult:
        cor_type = CoronaryType(coronary_type)
        cor_mesh = self.origin_data[cor_type].mesh_original.copy()
        mesh_points = torch.from_numpy(cor_mesh.points).to(self.device, torch.float32)
        mesh_points += self.rbf(phase, mesh_points)     # moving in world coordinate
        cor_mesh.points = mesh_points.cpu().numpy()

        assert self.origin_data[cor_type].artifacts_original is None

        ori_cor = self.origin_data[cor_type]
        
        return DataReaderResult(
            phase=phase,
            cavity=self.origin_data.cavity,
            coronary=Coronary(
                type=cor_type,
                volume=ori_cor.volume,
                label=ori_cor.label,
                centering_coord=ori_cor.centering_coord,
                mesh_original=cor_mesh,    # deformed mesh
                artifacts_original=None,
            )
        )
    
    @torch.no_grad()
    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        return self.get_combined_data(CardiacPhase(0), coronary_type)

    @property
    def lca_centering_affine(self) -> np.ndarray:
        return self.origin_data.lca.centering_affine
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        return self.origin_data.rca.centering_affine
    
    @property
    def volume_size(self) -> tuple[int, int, int]:
        return self._origin_volume_size
    
    @property
    def volume_affine(self) -> np.ndarray:
        return self._origin_volume_affine

    @property
    def meta(self) -> dict[str, Any]:
        source_files = {
            "cavity_nii": str(self.cavity_nii),
            "coronary_nii": str(self.coronary_nii),
        }
        return {
            "ssm_reader": self.ssm_reader.meta,
            "n_phases": self.n_phases,
            "centering_mode": self.centering_mode,
            "volume_size": self.volume_size,
            "volume_affine": self.volume_affine.tolist(),
            "source_files": source_files,
            "attenuation_transition": self.attenuation_transition,
            "contrast_simulator": self.contrast_simulator.meta if self.contrast_simulator is not None else None,
            "artifact_manager": self.artifact_manager.config.meta if self.artifact_manager is not None else None,
        }