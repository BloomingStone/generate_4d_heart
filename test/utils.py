from pathlib import Path
from enum import StrEnum
from functools import lru_cache
import warnings


from generate_4d_heart.rotate_dsa.data_reader import (
    VolumesReader, VolumeDVFReader, StaticVolumeReader, RBFReader,
    CoronaryBoundLVLinearEnhancer, CoronaryBoundLVEnhancer, StaticLabelReader, DataReader, IdentityMovementEnhancer
)
from generate_4d_heart.rotate_dsa.contrast_simulator import (
    MultipliContrast, ThresholdMultipliContrast, IdentityContrast, 
    SimplePreprocessContrast, ContrastSimulator, FlowContrast
)


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
    
    @classmethod
    def to_list(cls) -> list[str]:
        return [reader.value for reader in ReaderName]
    
    @classmethod
    def to_tuple(cls) -> tuple[str, ...]:
        return tuple(cls.to_list())


class SimulatorName(StrEnum):
    MULTIPLI_CONTRAST = "multipli_contrast"
    THRESHOLD_MULTIPLI_CONTRAST = "threshold_multipli_contrast"
    IDENTITY_CONTRAST = "identity_contrast"
    SIMPLE_PREPROCESS_CONTRAST = "simple_preprocess_contrast"
    FLOW_CONTRAST = "flow_contrast"
    
    @classmethod
    def to_list(cls) -> list[str]:
        return [sim.value for sim in SimulatorName]
    
    @classmethod
    def to_tuple(cls) -> tuple[str, ...]:
        return tuple(cls.to_list())



@lru_cache(maxsize=5)
def simulator_factory(
    simulator_name: SimulatorName
) -> ContrastSimulator:
    match simulator_name:
        case SimulatorName.MULTIPLI_CONTRAST:
            return MultipliContrast()
        case SimulatorName.THRESHOLD_MULTIPLI_CONTRAST:
            return ThresholdMultipliContrast()
        case SimulatorName.IDENTITY_CONTRAST:
            return IdentityContrast()
        case SimulatorName.SIMPLE_PREPROCESS_CONTRAST:
            return SimplePreprocessContrast()
        case SimulatorName.FLOW_CONTRAST:
            return FlowContrast()
        case _:
            raise ValueError(f"Unsupported simulator name: {simulator_name}")


def reader_factory(
    simulator_name: SimulatorName,
    reader_name: ReaderName,
    data_root_dir: Path = TEST_DATA_ROOT_DIR
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
        
    return simulator, reader


# --- DRRs
