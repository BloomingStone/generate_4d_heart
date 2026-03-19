import pytest
import numpy as np
import pyvista as pv
import nibabel as nib
from pathlib import Path
from generate_4d_heart.ssm.ssm import SSMReader
import logging
import pyvista as pv
from pyvista import CameraPosition

@pytest.fixture(scope="module")
def test_data_dir():
    # 假设测试数据路径
    return Path(__file__).parent.parent / "test" / "test_data" / "volume_with_dvf"

@pytest.fixture(scope="module")
def output_dir():
    return Path(__file__).parent.parent / "test" / "output" / "ssm_test"

@pytest.fixture(scope="module")
def cavity_img(test_data_dir):
    return nib.load(test_data_dir / "cavity.nii.gz")

@pytest.fixture(scope="module")
def ssm_reader():
    return SSMReader()

def test_ssm_load_and_deform(cavity_img, ssm_reader, output_dir):
    # 测试SSM加载和形变
    result = ssm_reader.load(cavity_img)
    assert hasattr(result, "landmark_with_motion")
    assert hasattr(result, "deformed_cavities")
    assert len(result.deformed_cavities) == ssm_reader.n_phases

    # 导出VTK和NII
    out_dir = output_dir / "ssm_out"
    result.save_vtk(out_dir)
    result.save_nii(cavity_img, out_dir)
    assert (out_dir / "landmark_with_motion.vtk").exists()
    assert (out_dir / "landmark_with_motion.nii.gz").exists()

def test_ssm_gif_visualization(cavity_img, ssm_reader, output_dir):
    # 生成多期相的gif动画，人工检查

    result = ssm_reader.load(cavity_img)
    meshes = result.deformed_cavities

    plotter = pv.Plotter(off_screen=True)
    plotter.camera_position = CameraPosition(
        (-68, 104, 344), (0, 0, 0), (0.75, -0.57, 0.32)
    )
    gif_path = output_dir / "ssm_motion.gif"
    plotter.open_gif(str(gif_path), fps=10)
    c = np.array(meshes[0].center)
    for mesh in meshes:
        plotter.clear()
        plotter.add_mesh(mesh.translate(-c), scalars="label", show_edges=True, opacity=0.5)
        plotter.write_frame()
    plotter.close()
    assert gif_path.exists()

def test_ssm_load_with_numpy(cavity_img, ssm_reader):
    # 用numpy array和affine测试
    arr = cavity_img.get_fdata().astype(np.uint8)
    affine = cavity_img.affine
    result = ssm_reader.load(arr, affine)
    assert hasattr(result, "deformed_cavities")
    assert len(result.deformed_cavities) == ssm_reader.n_phases

def test_ssm_load_with_path(cavity_img, ssm_reader, tmp_path):
    # 用文件路径测试
    nii_path = tmp_path / "cavity_tmp.nii.gz"
    nib.save(cavity_img, nii_path)
    result = ssm_reader.load(nii_path)
    assert hasattr(result, "deformed_cavities")
    assert len(result.deformed_cavities) == ssm_reader.n_phases

# 可选：测试不同主成分数量和motion_multiplier
# @pytest.mark.parametrize("num_components", [1, 2, 3])
# @pytest.mark.parametrize("motion_multiplier", [None, 0.8, 1.2])
# def test_ssm_load_parametric(cavity_img, ssm_reader, num_components, motion_multiplier):
#     result = ssm_reader.load(
#         cavity_img, num_components_used=num_components, motion_multiplier=motion_multiplier
#     )
#     assert hasattr(result, "deformed_cavities")
#     assert len(result.deformed_cavities) == ssm_reader.n_phases