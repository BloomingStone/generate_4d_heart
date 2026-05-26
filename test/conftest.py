from pathlib import Path
import warnings

import pytest

from generate_4d_heart.rotate_dsa.contrast_simulator import StaticIodineContrast, ThresholdIodineContrast, IdentityContrast, SimplePreprocessContrast, ContrastSimulator, FlowContrast
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters
from utils import TEST_ROOT_DIR, TEST_DATA_ROOT_DIR, OUTPUT_ROOT_DIR

@pytest.fixture(scope="session", autouse=True)
def test_root_dir() -> Path:
    print(f"Test root directory: {test_root_dir}")
    return TEST_ROOT_DIR

@pytest.fixture(scope="session", autouse=True)
def test_data_root_dir(
    test_root_dir: Path
) -> Path:
    test_data_root_dir = test_root_dir / "test_data"
    
    if test_data_root_dir != TEST_DATA_ROOT_DIR:
        warnings.warn(f"Test data root directory is set to {test_data_root_dir}, which is different from the default {TEST_DATA_ROOT_DIR}. Please make sure this is intentional.")
        
    assert test_data_root_dir.exists(), f"Test data root directory does not exist: {test_data_root_dir}"
    print(f"Test data root directory: {test_data_root_dir}")
    return test_data_root_dir

@pytest.fixture(scope="session", autouse=True)
def output_root_dir(
    test_root_dir: Path
) -> Path:
    output_root_dir = test_root_dir / "output"
    output_root_dir.mkdir(exist_ok=True)
    
    if output_root_dir != OUTPUT_ROOT_DIR:
        warnings.warn(f"Output root directory is set to {output_root_dir}, which is different from the default {OUTPUT_ROOT_DIR}. Please make sure this is intentional.")
    
    print(f"Output root directory: {output_root_dir}")
    return output_root_dir

@pytest.fixture(scope="session", autouse=True)
def volumes_dir(test_data_root_dir: Path) -> Path:
    p = test_data_root_dir / "volumes"
    assert p.exists(), f"Volumes directory does not exist: {p}"
    print(f"Volumes directory: {p}")
    return p

@pytest.fixture(scope="session", autouse=True)
def volume_dvf_dir(test_data_root_dir: Path) -> Path:
    p = test_data_root_dir / "volume_with_dvf"
    assert p.exists(), f"Volume with DVF directory does not exist: {p}"
    print(f"Volume with DVF directory: {p}")
    return p

# --- Contrast simulators

@pytest.fixture(scope="session", autouse=True)
def multipli_contrast() -> ContrastSimulator:
    print("Setting up StaticIodineContrast for the entire test session...")
    return StaticIodineContrast()

@pytest.fixture(scope="session", autouse=True)
def threshold_multipli_contrast() -> ContrastSimulator:
    print("Setting up ThresholdIodineContrast for the entire test session...")
    return ThresholdIodineContrast()

@pytest.fixture(scope="session", autouse=True)
def identity_contrast() -> ContrastSimulator:
    print("Setting up IdentityContrast for the entire test session...")
    return IdentityContrast()

@pytest.fixture(scope="session", autouse=True)
def simple_preprocess_contrast() -> ContrastSimulator:
    print("Setting up SimplePreprocessContrast for the entire test session...")
    return SimplePreprocessContrast()

@pytest.fixture(scope="session", autouse=True)
def flow_contrast() -> ContrastSimulator:
    print("Setting up FlowContrast for the entire test session...")
    return FlowContrast()

@pytest.fixture(scope="session", autouse=True)
def simple_drr() -> TorchDRR:
    rotate_cfg = RotatedParameters(
        total_frame=20,  # 减少帧数
        fps=7,           # 降低帧率, 使用质数避免与total_frame有过多公约数
        angular_velocity = 20,  # 降低旋转速度
    )
    return TorchDRR(rotate_cfg=rotate_cfg)


@pytest.fixture(scope="session", autouse=True)
def full_drr() -> TorchDRR:
    return TorchDRR(
        rotate_cfg=RotatedParameters(
            total_frame=120,
        )
    )

