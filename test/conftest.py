from pathlib import Path
import shutil
import warnings

import pytest

from generate_4d_heart.rotate_dsa.contrast_simulator import StaticIodineContrast, ThresholdIodineContrast, IdentityContrast, Hu2MuContrast, ContrastSimulator, FlowContrast
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters, CArmGeometry
from utils import TEST_ROOT_DIR, TEST_DATA_ROOT_DIR, OUTPUT_ROOT_DIR
from generate_4d_heart.rotate_dsa.data_reader import CenteringMode


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


@pytest.fixture(scope="session", autouse=True)
def simple_drr() -> TorchDRR:
    rotate_cfg = RotatedParameters(
        total_frame=40,  # 减少帧数
        fps=10,           # 降低帧率
        angular_velocity = 30,  # 降低旋转速度
    )
    return TorchDRR(rotate_cfg=rotate_cfg)

@pytest.fixture(scope="session", autouse=True)
def pig_data_geom() -> CArmGeometry:
    return CArmGeometry(
        sdd=1200.0,
        sod=760.0,
        height=474,
        width=648,
        delx=0.616,
        dely=0.616,
    )


@pytest.fixture(scope="session", autouse=True)
def pig_data_drr(
    pig_data_geom: CArmGeometry
) -> TorchDRR:
    rotate_cfg = RotatedParameters(
        total_frame=20, #402,  
        fps=15,         #80，      
        angular_velocity = 47.986,  
        alpha_start=-119.35
    )

    return TorchDRR(
        c_arm_cfg=pig_data_geom,
        rotate_cfg=rotate_cfg
    )

@pytest.fixture(scope="session", autouse=True)
def single_data_geom() -> CArmGeometry:
    return CArmGeometry(
        height=474,
        width=648,
        delx=0.3,
        dely=0.6,
    )


@pytest.fixture(scope="session", autouse=True)
def single_data_drr(
    single_data_geom: CArmGeometry
) -> TorchDRR:
    rotate_cfg = RotatedParameters(
        total_frame=1,
    )
    return TorchDRR(
        c_arm_cfg=single_data_geom,
        rotate_cfg=rotate_cfg
    )


from generate_4d_heart.rotate_dsa.artifacts import (
    ArtifactManager, ArtifactManagerConfig,
    GuideWireConfig, MarkerConfig, StentConfig,
)

@pytest.fixture(scope="session")
def artifact_manager() -> ArtifactManager:
    """ArtifactManager with guide wire + markers + stent."""
    return ArtifactManager(ArtifactManagerConfig(
        guide_wire=GuideWireConfig(radius_mm=0.3),
        markers=MarkerConfig(distance_mm=8.0, radius_mm=0.4, margin_ratio=0.8),
        stent=StentConfig(radius_mm=1., strut_radius_mm=0.15, n_wires=6),
    ))


@pytest.fixture(scope="session", autouse=True)
def full_drr() -> TorchDRR:
    return TorchDRR(
        rotate_cfg=RotatedParameters(
            total_frame=120,
        )
    )


