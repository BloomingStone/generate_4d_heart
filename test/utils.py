from pathlib import Path
from enum import StrEnum

from typing import Literal


from generate_4d_heart.rotate_dsa.data_reader import (
    VolumesReader, VolumeDVFReader, StaticVolumeReader, RBFReader,
    CoronaryBoundLVLinearEnhancer, CoronaryBoundLVEnhancer, StaticLabelReader, DataReader, IdentityMovementEnhancer
)
from generate_4d_heart.rotate_dsa.contrast_simulator import (
    StaticIodineContrast, ThresholdIodineContrast, IdentityContrast, 
    Hu2MuContrast, ContrastSimulator, FlowContrast, ClMarkerSimulator,
    CtaResetContrast
)
from generate_4d_heart.types import CoronaryType


TEST_ROOT_DIR = Path(__file__).parent
TEST_DATA_ROOT_DIR = TEST_ROOT_DIR / "test_data"
OUTPUT_ROOT_DIR = TEST_ROOT_DIR / "output"



# --- Dynamic reader functory

class ReaderName(StrEnum):
    VOLUMES_READER = "volumes_reader"
    VOLUME_DVF_READER = "volume_dvf_reader"
    RBF_READER = "rbf_reader"
    STATIC_VOLUME_READER = "static_volume_reader"
    STATIC_LABEL_READER = "static_label_reader"


class SimulatorName(StrEnum):
    STATIC_IODINE_CONTRAST = "static_iodine_contrast"
    THRESHOLD_IODINE_CONTRAST = "threshold_iodine_contrast"
    IDENTITY_CONTRAST = "identity_contrast"
    FLOW_CONTRAST = "flow_contrast"
    CL_MARKER_CONTRAST = "cl_marker_contrast"
    HU2MU_CONTRAST = "hu2mu_contrast"
    CTA_RESET_CONTRAST = "cta_reset_contrast"


def simulator_factory(
    simulator_name: SimulatorName
) -> ContrastSimulator:
    match simulator_name:
        case SimulatorName.STATIC_IODINE_CONTRAST:
            return StaticIodineContrast()
        case SimulatorName.THRESHOLD_IODINE_CONTRAST:
            return ThresholdIodineContrast()
        case SimulatorName.IDENTITY_CONTRAST:
            return IdentityContrast()
        case SimulatorName.FLOW_CONTRAST:
            return FlowContrast()
        case SimulatorName.CL_MARKER_CONTRAST:
            return ClMarkerSimulator()
        case SimulatorName.HU2MU_CONTRAST:
            return Hu2MuContrast()
        case SimulatorName.CTA_RESET_CONTRAST:
            return CtaResetContrast()
        case _:
            raise ValueError(f"Unsupported simulator name: {simulator_name}")


def reader_factory(
    simulator_name: SimulatorName,
    reader_name: ReaderName,
    data_root_dir: Path = TEST_DATA_ROOT_DIR,
    cororonary_need_load: set[CoronaryType]|Literal["LCA", "RCA"]|None = None
) -> tuple[ContrastSimulator, DataReader]:
    volumes_dir = data_root_dir / "volumes"
    volume_dvf_dir = data_root_dir / "volume_with_dvf"
    
    simulator = simulator_factory(simulator_name)
    
    match reader_name:
        case ReaderName.VOLUMES_READER:
            reader = VolumesReader(
                image_dir=volumes_dir / "image",
                cavity_dir=volumes_dir / "cavity",
                coronary_dir=volumes_dir / "coronary",
                contrast_simulator=simulator
            )
        case ReaderName.VOLUME_DVF_READER:
            reader = VolumeDVFReader(
                volume_nii=volume_dvf_dir / "image.nii.gz",
                cavity_nii=volume_dvf_dir / "cavity.nii.gz",
                coronary_nii=volume_dvf_dir / "coronary.nii.gz",
                dvf_dir=volume_dvf_dir / "dvf",
                roi_json=volume_dvf_dir / "Normal_01.json",
                movement_enhancer=CoronaryBoundLVLinearEnhancer(),
                contrast_simulator=simulator
            )
        case ReaderName.RBF_READER:
            reader = RBFReader(
                volume_nii=volume_dvf_dir / "image.nii.gz",
                cavity_nii=volume_dvf_dir / "cavity.nii.gz",
                coronary_nii=volume_dvf_dir / "coronary.nii.gz",
                contrast_simulator=simulator
            )
        case ReaderName.STATIC_VOLUME_READER:
            reader = StaticVolumeReader(
                volume_path=volume_dvf_dir / "image.nii.gz",
                cavity_path=volume_dvf_dir / "cavity.nii.gz",
                coronary_path=volume_dvf_dir / "coronary.nii.gz",
                contrast_simulator=simulator
            )
        case ReaderName.STATIC_LABEL_READER:
            reader = StaticLabelReader(
                cavity_path=volume_dvf_dir / "cavity.nii.gz",
                coronary_path=volume_dvf_dir / "coronary.nii.gz",
                contrast_simulator=simulator
            )
        case _:
            raise ValueError(f"Unsupported reader name: {reader_name}")
    
    if cororonary_need_load is not None:
        if isinstance(cororonary_need_load, str):
            cororonary_need_load = {CoronaryType(cororonary_need_load)}
        reader.coronary_need_load = cororonary_need_load
    return simulator, reader


# --- DRRs
