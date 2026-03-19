import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from generate_4d_heart.roi import ROI
import torch

import matplotlib.pyplot as plt


@pytest.fixture(scope="module")
def cavity_path():
    return Path(__file__).parent / "test_data" / "volume_with_dvf" / "cavity.nii.gz"

@pytest.fixture(scope="module")
def cavity_img(cavity_path):
    return nib.load(str(cavity_path))

@pytest.fixture(scope="module")
def output_dir():
    return Path(__file__).parent.parent / "test" / "output" / "roi_test"

def test_roi_affine_and_shape_consistency(cavity_img):
    roi = ROI.get_from_cavity(cavity_img, padding=8)
    # 手动计算裁剪框
    crop_box = roi.get_crop_box()
    (x0, x1), (y0, y1), (z0, z1) = crop_box
    manual_shape = (x1 - x0, y1 - y0, z1 - z0)
    assert manual_shape == roi.shape_after_crop

    # 检查缩放后shape
    expected_zoom_shape = tuple((np.array(manual_shape) * roi.get_zoom_rate()).astype(int))
    assert expected_zoom_shape == roi.shape_after_crop_and_zoom

    # 检查affine裁剪后
    orig_affine = roi.original_affine
    T = np.eye(4)
    T[:3, 3] = np.array([x0, y0, z0])
    manual_affine_crop = orig_affine @ T
    np.testing.assert_allclose(manual_affine_crop, roi.affine_after_crop)

    # 检查affine裁剪+缩放后
    R = np.eye(4)
    R[:3, :3] = np.diag(1 / roi.get_zoom_rate())
    manual_affine_crop_zoom = manual_affine_crop @ R
    np.testing.assert_allclose(manual_affine_crop_zoom, roi.affine_after_crop_and_zoom)

def test_roi_to_dict_and_from_dict(cavity_img):
    roi = ROI.get_from_cavity(cavity_img, padding=8)
    roi_dict = roi.to_dict()
    roi2 = ROI.from_dict(roi_dict)
    assert np.allclose(roi.get_crop_box(), roi2.get_crop_box())
    assert np.allclose(roi.get_zoom_rate(), roi2.get_zoom_rate())
    assert roi.original_shape == roi2.original_shape
    assert np.allclose(roi.original_affine, roi2.original_affine)

def test_roi_recover_cropped_tensor(cavity_img):
    roi = ROI.get_from_cavity(cavity_img, padding=8)
    cropped = roi.crop_on_data(torch.ones(roi.original_shape))
    recovered = roi.recover_cropped_tensor(cropped)
    # 检查恢复后非零区域与crop_box一致
    (x0, x1), (y0, y1), (z0, z1) = roi.get_crop_box()
    mask = torch.zeros(roi.original_shape)
    mask[x0:x1, y0:y1, z0:z1] = 1
    assert torch.all(recovered == mask)

def test_roi_crop_and_zoom(cavity_img, output_dir):
    # 创建ROI对象
    roi = ROI.get_from_cavity(cavity_img, padding=8)
    # 裁剪
    cropped_img = roi.crop(cavity_img)
    cropped_data = cropped_img.get_fdata()
    # 裁剪+缩放
    cropped_zoom_img = roi.crop_zoom(cavity_img, is_label=True)
    cropped_zoom_data = cropped_zoom_img.get_fdata()

    # 可视化原图、裁剪后、裁剪缩放后
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    orig_data = cavity_img.get_fdata()
    slices = [s//2 for s in orig_data.shape]
    crop_slices = [s//2 for s in cropped_data.shape]
    zoom_slices = [s//2 for s in cropped_zoom_data.shape]

    for i, axis in enumerate([0, 1, 2]):
        axs[0, i].imshow(np.take(orig_data, slices[axis], axis=axis), cmap="gray")
        axs[0, i].set_title(f"Original axis={axis}")
        axs[1, i].imshow(np.take(cropped_data, crop_slices[axis], axis=axis), cmap="gray")
        axs[1, i].set_title(f"Cropped axis={axis}")
        axs[2, i].imshow(np.take(cropped_zoom_data, zoom_slices[axis], axis=axis), cmap="gray")
        axs[2, i].set_title(f"Cropped+Zoom axis={axis}")

    plt.tight_layout()
    out_png = output_dir / "roi_crop_zoom_vis.png"
    plt.savefig(out_png)
    plt.close()
    assert out_png.exists()