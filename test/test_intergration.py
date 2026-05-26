from typing import Literal
from pathlib import Path
import shutil

import pytest

from generate_4d_heart.rotate_dsa import RotateDSA
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters

from utils import reader_factory, ReaderName, SimulatorName


@pytest.mark.parametrize("reader_name", ReaderName.to_tuple())
@pytest.mark.parametrize("simulator_name", (
    str(SimulatorName.MULTIPLI_CONTRAST), str(SimulatorName.FLOW_CONTRAST)
))
@pytest.mark.parametrize("coronary_type", ["LCA", "RCA"])
def test_rotate_dsa_integration(
    output_root_dir: Path,
    simple_drr: TorchDRR,
    reader_name: str,
    simulator_name: str,
    coronary_type: Literal["LCA", "RCA"],
):
    """测试完整的 RotateDSA 集成流程"""
    _, reader = reader_factory(SimulatorName(simulator_name), ReaderName(reader_name))
    
    # 创建 RotateDSA 实例
    dsa = RotateDSA(
        reader=reader,
        drr=simple_drr
    )
    
    # 测试保存功能
    output_dir = output_root_dir / "intergration" / f"{reader_name}_{simulator_name}_{coronary_type}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    frames, labels, depth_map = dsa.run_and_save(
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
    assert (output_dir / "rotate_dsa.tif").exists()
    assert (output_dir / "rotate_dsa.gif").exists()
    assert (output_dir / "rotate_dsa.json").exists()


@pytest.mark.parametrize(
    "reader_name, coronary_type",
    (
        ("volumes_reader", "LCA"),
        ("volume_dvf_reader", "LCA"),
        ("static_volume_reader", "LCA"),
        ("rbf_reader", "LCA"),
        ("rbf_reader", "RCA"),      # RBFReader will be fully tested on both LCA and RCA
        ("static_label_reader", "LCA"),  # StaticLabelReader only has LCA data
    )
)
@pytest.mark.parametrize("simulator_name", (
    str(SimulatorName.MULTIPLI_CONTRAST), 
    str(SimulatorName.FLOW_CONTRAST)
))
def test_rotate_dsa_integration_full(
    output_root_dir: Path,
    full_drr: TorchDRR,
    reader_name: str,
    simulator_name: str,
    coronary_type: Literal["LCA", "RCA"],
):
    """测试完整的 RotateDSA 集成流程 并生成完整数据"""
    _, reader = reader_factory(SimulatorName(simulator_name), ReaderName(reader_name))
    
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