from typing import Literal
from pathlib import Path
import shutil

import pytest

from generate_4d_heart.rotate_dsa import RotateDSA
from generate_4d_heart.rotate_dsa.artifacts.manager import StentConfig
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters
from generate_4d_heart.rotate_dsa.artifacts import (
    ArtifactManager, ArtifactManagerConfig, GuideWireConfig, MarkerConfig,
)
from generate_4d_heart.rotate_dsa.data_reader import RBFReader
from generate_4d_heart.rotate_dsa.contrast_simulator import IdentityContrast

from utils import reader_factory, ReaderName, SimulatorName, TEST_DATA_ROOT_DIR, simulator_factory


@pytest.mark.parametrize("reader_name", (ReaderName.RBF_READER, ReaderName.VOLUME_DVF_READER, ReaderName.VOLUMES_READER, ReaderName.STATIC_VOLUME_READER))
@pytest.mark.parametrize("simulator_name", (
    SimulatorName.STATIC_IODINE_CONTRAST, SimulatorName.FLOW_CONTRAST, SimulatorName.HU2MU_CONTRAST
))
@pytest.mark.parametrize("coronary_type", ["LCA", "RCA"])
def test_integ(
    output_root_dir: Path,
    simple_drr: TorchDRR,
    reader_name: ReaderName,
    simulator_name: SimulatorName,
    coronary_type: Literal["LCA", "RCA"],
):
    """测试完整的 RotateDSA 集成流程"""
    _, reader = reader_factory(SimulatorName(simulator_name), ReaderName(reader_name), cororonary_need_load=coronary_type)
    
    # 创建 RotateDSA 实例
    dsa = RotateDSA(
        reader=reader,
        drr=simple_drr
    )
    
    # 测试保存功能
    output_dir = output_root_dir / "intergration" / f"{reader_name}_{simulator_name}_{coronary_type}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dsa.run_and_save(
        output_dir=output_dir,
        coronary_type=coronary_type,
        gif_fps=10
    )
    
    geometry_json = dsa.get_geometry_json(coronary_type)
    
    # 验证几何信息
    assert "frames" in geometry_json
    assert "c_arm_geometry" in geometry_json
    assert "rotate_parameters" in geometry_json
    assert len(geometry_json["frames"]) == simple_drr.rotate_cfg.total_frame
    
    # 验证每帧都有角度和相位信息
    for frame_info in geometry_json["frames"]:
        assert "frame" in frame_info
        assert "phase" in frame_info
        assert "alpha_degree" in frame_info
        assert "beta_degree" in frame_info
        alpha_degree = frame_info["alpha_degree"]
        assert isinstance(alpha_degree, float)
        beta_degree = frame_info["beta_degree"]
        assert isinstance(beta_degree, float)
        assert isinstance(frame_info["phase"], (int, float))
    
    # 验证输出文件
    assert (output_dir / "rotate_dsa_raw.tif").exists()
    assert (output_dir / "rotate_dsa_raw.gif").exists()
    assert (output_dir / "rotate_dsa.gif").exists()
    assert (output_dir / "rotate_dsa.json").exists()

@pytest.mark.parametrize("reader_name", (ReaderName.STATIC_LABEL_READER,))
@pytest.mark.parametrize("simulator_name", (SimulatorName.IDENTITY_CONTRAST,))
@pytest.mark.parametrize("coronary_type", ["LCA", "RCA"])
def test_integ_only_label(
    output_root_dir: Path,
    simple_drr: TorchDRR,
    reader_name: ReaderName,
    simulator_name: SimulatorName,
    coronary_type: Literal["LCA", "RCA"],
):
    test_integ(
        output_root_dir=output_root_dir,
        simple_drr=simple_drr,
        reader_name=reader_name,
        simulator_name=simulator_name,
        coronary_type=coronary_type
    )


@pytest.mark.parametrize(
    "reader_name, coronary_type",
    (
        (ReaderName.VOLUMES_READER, "LCA"),
        (ReaderName.VOLUME_DVF_READER, "LCA"),
        (ReaderName.STATIC_VOLUME_READER, "LCA"),
        (ReaderName.RBF_READER, "LCA"),
        (ReaderName.RBF_READER, "RCA"),      # RBFReader will be fully tested on both LCA and RCA
        (ReaderName.STATIC_LABEL_READER, "LCA"),  # StaticLabelReader only has LCA data
    )
)
@pytest.mark.parametrize("simulator_name", (
    SimulatorName.STATIC_IODINE_CONTRAST, 
    SimulatorName.FLOW_CONTRAST
))
def test_integ_full(
    output_root_dir: Path,
    full_drr: TorchDRR,
    reader_name: ReaderName,
    simulator_name: SimulatorName,
    coronary_type: Literal["LCA", "RCA"],
):
    """测试完整的 RotateDSA 集成流程 并生成完整数据"""
    _, reader = reader_factory(SimulatorName(simulator_name), ReaderName(reader_name), cororonary_need_load=coronary_type)
    
    dsa = RotateDSA(
        reader=reader,
        drr=full_drr
    )
    
    output_dir = output_root_dir / "intergration_full" / f"{reader_name}_{simulator_name}_{coronary_type}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dsa.run_and_save(
        output_dir=output_dir,
        coronary_type=coronary_type
    )


@pytest.mark.parametrize("simulator_name", (
    SimulatorName.CTA_RESET_CONTRAST, SimulatorName.STATIC_IODINE_CONTRAST, SimulatorName.FLOW_CONTRAST
))
@pytest.mark.parametrize("coronary_type", ["LCA", "RCA"])
def test_integ_artifacts(
    output_root_dir: Path,
    simple_drr: TorchDRR,
    volume_dvf_dir: Path,
    simulator_name: SimulatorName,
    coronary_type: Literal["LCA", "RCA"],
    artifact_manager: ArtifactManager,
    comment: str = ""
):
    """RBFReader with artifact_manager + RotateDSA integration."""
    reader = RBFReader(
        volume_nii=volume_dvf_dir / "image.nii.gz",
        cavity_nii=volume_dvf_dir / "cavity.nii.gz",
        coronary_nii=volume_dvf_dir / "coronary.nii.gz",
        contrast_simulator=simulator_factory(simulator_name),
        artifact_manager=artifact_manager,
    )

    dsa = RotateDSA(reader=reader, drr=simple_drr)

    output_dir = output_root_dir / "intergration" / "artifacts" / f"rbf_reader_{simulator_name}_{coronary_type}{comment}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dsa.run_and_save(output_dir=output_dir, coronary_type=coronary_type, gif_fps=10)

    # Verify output files exist
    assert (output_dir / "rotate_dsa_raw.tif").exists()
    assert (output_dir / "rotate_dsa.json").exists()
    if dsa.drr.rotate_cfg.total_frame > 60:
        assert (output_dir / "rotate_dsa_raw.mp4").exists()
        assert (output_dir / "rotate_dsa.mp4").exists()
        assert (output_dir / "label.mp4").exists()
        assert (output_dir / "depth_map.mp4").exists()
    else:
        assert (output_dir / "rotate_dsa_raw.gif").exists()
        assert (output_dir / "rotate_dsa.gif").exists()
        assert (output_dir / "label.gif").exists()
        assert (output_dir / "depth_map.gif").exists()


@pytest.mark.parametrize("simulator_name", (
    SimulatorName.CTA_RESET_CONTRAST, SimulatorName.STATIC_IODINE_CONTRAST, SimulatorName.FLOW_CONTRAST
))
@pytest.mark.parametrize("coronary_type", ["LCA"])
def test_integ_artifacts_pig_drr(
    output_root_dir: Path,
    pig_data_drr: TorchDRR,
    volume_dvf_dir: Path,
    simulator_name: SimulatorName,
    coronary_type: Literal["LCA", "RCA"],
    artifact_manager: ArtifactManager
):
    """RBFReader with artifact_manager + RotateDSA integration."""
    test_integ_artifacts(
        output_root_dir=output_root_dir,
        simple_drr=pig_data_drr,
        volume_dvf_dir=volume_dvf_dir,
        simulator_name=simulator_name,
        coronary_type=coronary_type,
        artifact_manager=artifact_manager,
        comment="_pig_data"
    )



@pytest.mark.parametrize("simulator_name", (
    SimulatorName.HU2MU_CONTRAST, SimulatorName.STATIC_IODINE_CONTRAST, SimulatorName.CTA_RESET_CONTRAST, SimulatorName.FLOW_CONTRAST
))
def test_integ_artifacts_pig_drr_chest_background(
    output_root_dir: Path,
    pig_data_drr: TorchDRR,
    simulator_name: SimulatorName,
    artifact_manager: ArtifactManager,
):
    """Integration test with chest CTA data + background CT, using pig_data_drr."""
    chest_dir = TEST_DATA_ROOT_DIR / "chest_data"
    assert chest_dir.exists(), f"chest_data directory not found: {chest_dir}"

    reader = RBFReader(
        volume_nii=chest_dir / "image.nii.gz",
        cavity_nii=chest_dir / "cavity.nii.gz",
        coronary_nii=chest_dir / "coronary.nii.gz",
        contrast_simulator=simulator_factory(simulator_name),
        artifact_manager=artifact_manager,
        background_nii=chest_dir / "background.nii.gz",
        centering_mode = "image_center"
    )

    dsa = RotateDSA(reader=reader, drr=pig_data_drr)

    output_dir = output_root_dir / "intergration" / "artifacts" / f"pig_drr_chest_background_{simulator_name}_LCA"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dsa.run_and_save(output_dir=output_dir, coronary_type="LCA", gif_fps=10)

    # Verify output files exist
    assert (output_dir / "rotate_dsa_raw.tif").exists()
    assert (output_dir / "rotate_dsa.json").exists()
    if dsa.drr.rotate_cfg.total_frame > 60:
        assert (output_dir / "rotate_dsa_raw.mp4").exists()
        assert (output_dir / "rotate_dsa.mp4").exists()
        assert (output_dir / "label.mp4").exists()
        assert (output_dir / "depth_map.mp4").exists()
    else:
        assert (output_dir / "rotate_dsa_raw.gif").exists()
        assert (output_dir / "rotate_dsa.gif").exists()
        assert (output_dir / "label.gif").exists()
        assert (output_dir / "depth_map.gif").exists()


# @pytest.mark.parametrize("simulator_name", (
#     SimulatorName.CTA_RESET_CONTRAST, SimulatorName.STATIC_IODINE_CONTRAST
# ))
# def test_chest_data_background_as_image(
#     output_root_dir: Path,
#     pig_data_drr: TorchDRR,
#     simulator_name: SimulatorName,
#     artifact_manager: ArtifactManager,
# ):
#     """Integration test with chest CTA data + background CT, using pig_data_drr."""
#     chest_dir = TEST_DATA_ROOT_DIR / "chest_data"
#     assert chest_dir.exists(), f"chest_data directory not found: {chest_dir}"

#     reader = RBFReader(
#         volume_nii=chest_dir / "background.nii.gz",
#         cavity_nii=chest_dir / "cavity.nii.gz",
#         coronary_nii=chest_dir / "coronary.nii.gz",
#         contrast_simulator=simulator_factory(simulator_name),
#         artifact_manager=artifact_manager,
#     )

#     dsa = RotateDSA(reader=reader, drr=pig_data_drr)

#     output_dir = output_root_dir / "intergration" / "artifacts" / f"chest_background_{simulator_name}_LCA"
#     if output_dir.exists():
#         shutil.rmtree(output_dir)
#     dsa.run_and_save(output_dir=output_dir, coronary_type="LCA", gif_fps=10)

#     # Verify output files exist
#     assert (output_dir / "rotate_dsa_raw.tif").exists()
#     assert (output_dir / "rotate_dsa_raw.mp4").exists()
#     assert (output_dir / "rotate_dsa.mp4").exists()
#     assert (output_dir / "rotate_dsa.json").exists()
#     assert (output_dir / "label.mp4").exists()
#     assert (output_dir / "depth_map.mp4").exists()