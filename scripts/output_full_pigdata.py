from pathlib import Path
from enum import StrEnum
from typing import Literal
from pathlib import Path
import shutil
from dataclasses import dataclass

from cyclopts import run

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


def pig_data_geom() -> CArmGeometry:
    return CArmGeometry(
        sdd=1200.0,
        sod=760.0,
        height=474,
        width=648,
        delx=0.616,
        dely=0.616,
    )


def pig_data_drr(
    pig_data_geom: CArmGeometry
) -> TorchDRR:
    rotate_cfg = RotatedParameters.from_alpha_range(
        total_frame=402,
        alpha_start=-120.0,
        alpha_end=120.0,
        fps=80.0,
        beta_start=0.0,
        coordinate_system="RAS",
        parameterization="euler_angles",
        convention="ZXY"
    )
    return TorchDRR(
        c_arm_cfg=pig_data_geom,
        rotate_cfg=rotate_cfg
    )

def artifact_manager() -> ArtifactManager:
    """ArtifactManager with guide wire + markers + stent."""
    return ArtifactManager(ArtifactManagerConfig(
        guide_wire=GuideWireConfig(radius_mm=0.3),
        markers=MarkerConfig(distance_mm=8.0, radius_mm=0.4, margin_ratio=0.8),
        stent=StentConfig(radius_mm=1., strut_radius_mm=0.15, n_wires=6),
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
    centering_mode: CenteringMode = 'image_center'

def main(
    cfg: InputConfig = InputConfig(),
    output: Path = Path("output"),
    coronary_type: Literal["LCA", "RCA"] = "LCA",
    simulator_name: SimulatorName = SimulatorName.HU2MU_CONTRAST,
    artifact_manager: ArtifactManager = artifact_manager(),
    simple_drr: TorchDRR = pig_data_drr(pig_data_geom()),
):
    """RBFReader with artifact_manager + RotateDSA integration as pigdata settings."""
    output_dir = output / f"{simulator_name}_{coronary_type}"
    if output_dir.exists():
        ans = input("output_dir exists, overwrite it? (Y/n)")
        if ans.lower() != "y" and ans != "":
            raise ValueError("Output directory already exists and user chose not to overwrite it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reader = RBFReader(
        volume_nii=cfg.volume_nii,
        cavity_nii=cfg.cavity_nii,
        coronary_nii=cfg.coronary_nii,
        background_nii=cfg.background_nii,
        centering_mode=cfg.centering_mode,        
        contrast_simulator=simulator_factory(simulator_name),
        artifact_manager=artifact_manager,
    )

    dsa = RotateDSA(reader=reader, drr=simple_drr)

    dsa.run_and_save(output_dir=output_dir, coronary_type=coronary_type)

if __name__ == "__main__":
    run(main)
