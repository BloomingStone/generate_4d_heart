from pathlib import Path
from typing import Literal
import shutil

import pytest
import torch

from generate_4d_heart.rotate_dsa import RotateDSA
from generate_4d_heart.rotate_dsa.contrast_simulator import MultipliContrast, ThresholdMultipliContrast
from generate_4d_heart.rotate_dsa.data_reader import VolumesReader, VolumeDVFReader
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters

from utils import get_volumes_reader, get_volume_dvf_reader, output_root_dir


readers = {
    "volumes_reader": get_volumes_reader(),
    "volume_dvf_reader": get_volume_dvf_reader()
}

simulators = {
    "multipli_contrast": MultipliContrast(),
    "threshold_multipli_contrast": ThresholdMultipliContrast()
}

@pytest.mark.parametrize("reader_name, reader", readers.items())
@pytest.mark.parametrize("sim_name, simulator", simulators.items())
@pytest.mark.parametrize("coronary_type", ["LCA", "RCA"])
def test_rotate_dsa_integration(
    reader_name: str,
    reader: VolumesReader | VolumeDVFReader,
    sim_name: str,
    simulator: MultipliContrast | ThresholdMultipliContrast,
    coronary_type: Literal["LCA", "RCA"],
):
    """测试完整的 RotateDSA 集成流程"""
    # 配置 DRR 参数为测试模式（减少帧数和图像大小以加快测试速度）
    rotate_cfg = RotatedParameters(
        total_frame=20,  # 减少帧数
        fps=10,           # 降低帧率
    )
    
    drr = TorchDRR(rotate_cfg=rotate_cfg)
    
    # 创建 RotateDSA 实例
    dsa = RotateDSA(
        reader=reader,
        constrast_sim=simulator,
        drr=drr
    )
    
    # 测试保存功能
    output_dir = output_root_dir / "intergration" / f"{reader_name}_{sim_name}_{coronary_type}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    frames, geometry_json = dsa.run_and_save(
        output_dir=output_dir,
        base_name="test_rotate_dsa",
        coronary_type=coronary_type
    )
    
    # 验证输出格式
    assert isinstance(frames, torch.Tensor)
    assert frames.dtype == torch.uint8
    assert frames.shape == (rotate_cfg.total_frame, 1, *drr.image_size)
    
    # 验证几何信息
    assert "frames" in geometry_json
    assert "c_arm_geometry" in geometry_json
    assert "rotate_parameters" in geometry_json
    assert len(geometry_json["frames"]) == rotate_cfg.total_frame
    
    # 验证每帧都有角度和相位信息
    for frame_info in geometry_json["frames"]:
        assert "frame" in frame_info
        assert "phase" in frame_info
        assert "angle" in frame_info
        assert isinstance(frame_info["angle"], tuple) and len(frame_info["angle"]) == 3
        assert isinstance(frame_info["phase"], (int, float))
        assert isinstance(frame_info["R"], list) and len(frame_info["R"]) == 3 and len(frame_info["R"][0]) == 3
        assert isinstance(frame_info["T"], list) and len(frame_info["T"]) == 3
    
    # 验证输出文件
    assert (output_dir / "test_rotate_dsa.tif").exists()
    assert (output_dir / "test_rotate_dsa.gif").exists()
    assert (output_dir / "test_rotate_dsa.json").exists()