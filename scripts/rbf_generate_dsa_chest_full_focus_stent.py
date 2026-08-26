#!/usr/bin/env -S pixi run python

from pathlib import Path
from enum import StrEnum
from typing import Literal, Annotated
from pathlib import Path
from dataclasses import dataclass

from cyclopts import run, Parameter
import numpy as np

from generate_4d_heart.rotate_dsa.contrast_simulator import StaticIodineContrast, Hu2MuContrast, ContrastSimulator, FlowContrast, CtaResetContrast
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters, CArmGeometry
from generate_4d_heart.rotate_dsa.data_reader import CenteringMode
from generate_4d_heart.rotate_dsa import RotateDSA
from generate_4d_heart.rotate_dsa.artifacts.manager import StentConfig
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters
from generate_4d_heart.rotate_dsa.artifacts import (
    ArtifactManager, ArtifactManagerConfig, GuideWireConfig, MarkerConfig,
)
from generate_4d_heart.rotate_dsa.data_reader import RBFReader


def full_data_geom() -> CArmGeometry:
    return CArmGeometry(
        sdd=1200.0,
        sod=800.0,
        height=1024,
        width=1024,
        delx=0.154,
        dely=0.154,
    )


def full_data_drr(
    full_data_geom: CArmGeometry
) -> TorchDRR:
    rotate_cfg = RotatedParameters.from_alpha_range(
        total_frame=402,
        alpha_start=-100.0,
        alpha_end=100.0,
        fps=80.0,
        beta_start=0.0,
        coordinate_system="RAS",
        parameterization="euler_angles",
        convention="ZXY"
    )
    return TorchDRR(
        c_arm_cfg=full_data_geom,
        rotate_cfg=rotate_cfg,
        patch_size=128,
    )

def artifact_manager() -> ArtifactManager:
    """ArtifactManager with guide wire + markers + stent."""
    return ArtifactManager(ArtifactManagerConfig(
        guide_wire=GuideWireConfig(radius_mm=0.2),
        markers=MarkerConfig(distance_mm=19.0, radius_mm=0.5, start_pos_mm=30.0),
        stent=StentConfig(radius_mm=1.5, strut_radius_mm=0.15, n_wires=6),
    ))


class SimulatorName(StrEnum):
    STATIC_IODINE_CONTRAST = "static_iodine_contrast"
    FLOW_CONTRAST = "flow_contrast"
    HU2MU_CONTRAST = "hu2mu_contrast"
    CTA_RESET_CONTRAST = "cta_reset_contrast"

def simulator_factory(
    simulator_name: SimulatorName
) -> ContrastSimulator:
    match simulator_name:
        case SimulatorName.STATIC_IODINE_CONTRAST:
            return StaticIodineContrast()
        case SimulatorName.HU2MU_CONTRAST:
            return Hu2MuContrast()
        case SimulatorName.FLOW_CONTRAST:
            return FlowContrast()
        case SimulatorName.HU2MU_CONTRAST:
            return Hu2MuContrast()
        case SimulatorName.CTA_RESET_CONTRAST:
            return CtaResetContrast()
        case _:
            raise ValueError(f"Unsupported simulator name: {simulator_name}")

@dataclass
class InputConfig:
    volume_nii: Path = Path("test/test_data/chest_data/image.nii.gz")
    cavity_nii: Path = Path("test/test_data/chest_data/cavity.nii.gz")
    coronary_nii: Path = Path("test/test_data/chest_data/coronary.nii.gz")
    background_nii: Path = Path("test/test_data/chest_data/background.nii.gz")
    centering_mode: CenteringMode = 'coronary_center'

def main(
    cfg: InputConfig = InputConfig(),
    output: Path = Path("temp/output"),
    coronary_type: Literal["LCA", "RCA"] = "LCA",
    simulator_name: SimulatorName = SimulatorName.CTA_RESET_CONTRAST,
    artifact_manager: ArtifactManager = artifact_manager(),
    simple_drr: TorchDRR = full_data_drr(full_data_geom()),
    yes_to_all: Annotated[bool, Parameter(name='-y', help="Skip all confirmation prompts.")] = False
) -> None:
    """
    RBFReader with artifact_manager + RotateDSA integration as fulldata settings.
    """
    output_dir = output / f"{simulator_name}_{coronary_type}"
    if output_dir.exists():
        if yes_to_all:
            print(f"output_dir {output_dir} exists, overwriting it.")
        else:
            ans = input(f"output_dir {output_dir} exists, overwrite it? (Y/n)")
            if ans.lower() != "y" and ans != "":
                raise ValueError(f"Output directory {output_dir} already exists and user chose not to overwrite it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reader = RBFReader(
        volume_nii=cfg.volume_nii,
        cavity_nii=cfg.cavity_nii,
        coronary_nii=cfg.coronary_nii,
        background_nii=cfg.background_nii,
        centering_mode=cfg.centering_mode,        
        contrast_simulator=simulator_factory(simulator_name),
        artifact_manager=artifact_manager,
        transform_to_centering=np.array([0.0, 15.0, 0.0], dtype=np.float32)
    )

    dsa = RotateDSA(reader=reader, drr=simple_drr)

    dsa.run_and_save(output_dir=output_dir, coronary_type=coronary_type, 
                     phase_0_save_types=("coronary", "volume", "cavity", "cor-mesh", "cor-artifacts", "central-line", "meta"))
    

if __name__ == "__main__":
    run(main)
