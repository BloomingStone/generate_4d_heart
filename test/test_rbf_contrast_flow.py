from typing import Literal
import shutil

import pytest
import torch

from generate_4d_heart.rotate_dsa import RotateDSA
from generate_4d_heart.rotate_dsa.contrast_simulator import MultipliContrast, ThresholdMultipliContrast
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters
from generate_4d_heart.rotate_dsa.data_reader.rbf_reader import RBFReader
from generate_4d_heart.rotate_dsa.contrast_simulator.flow_contrast import FlowContrast

from utils import output_root_dir, get_reader, get_simulator, test_data_root_dir


@pytest.mark.parametrize("coronary_type", ["LCA", "RCA"])
def test_rbf_contrast_flow_integration(
    coronary_type: Literal["LCA", "RCA"],
):
    """测试完整的 RotateDSA 集成流程"""
    # 配置 DRR 参数为测试模式（减少帧数和图像大小以加快测试速度）
    rotate_cfg = RotatedParameters(
        total_frame=20,  # 减少帧数
        fps=5,           # 降低帧率
        angular_velocity = 20,  # 降低旋转速度
    )
    
    drr = TorchDRR(rotate_cfg=rotate_cfg)
    
    # 创建 RotateDSA 实例
    
    data_dir = test_data_root_dir / "volume_with_dvf"
    dsa = RotateDSA(
        reader=RBFReader(
            volume_nii=data_dir / "image.nii.gz",
            cavity_nii=data_dir / "cavity.nii.gz",
            coronary_nii=data_dir / "coronary.nii.gz",
            contrast_simulator=FlowContrast()
        ),
        drr=drr
    )
    
    # 测试保存功能
    output_dir = output_root_dir / "intergration" / f"rbf_contrast_flow_{coronary_type}"
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
    assert len(geometry_json["frames"]) == rotate_cfg.total_frame
    
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