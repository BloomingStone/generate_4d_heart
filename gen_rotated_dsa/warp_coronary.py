from monai.networks.blocks.warp import Warp
from monai.transforms import AsDiscrete # type: ignore
import torch
from torch import nn
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_dilation

def get_coronary_centerline_ddf(
        model: ShapeMorph,      # type: ignore
        dvf: torch.Tensor,
        cropped_image: torch.Tensor,
        cropped_coronary_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用形变场对图像进行变形，并将冠脉区域的形变替换为中心线的形变
    Args:
        model: ShapeMorph
        dvf: (B, C, W, H, D) 密集速度场
        image: (B, C, W, H, D) 图像
        coronary_mask: (W, H, D) 冠脉掩膜
    Returns:
        wrapped_image: (W, H, D) 变形后的图像
        wrapped_coronary: (W, H, D) 按中心线变形后的冠脉
        ddf: (C, W, H, D) 最终形变场
    """
    to_binary = AsDiscrete(threshold=0.5)
    dvf2ddf = model.dvf2ddf
    image_warp = Warp(mode='bilinear', padding_mode='zeros')
    warp_label = Warp(mode='nearest', padding_mode='zeros')
    ddf = dvf2ddf(dvf)
    assert isinstance(ddf, torch.Tensor)
    coronary_mask_np = binary_dilation(cropped_coronary_mask, iterations=5).astype(np.uint8)
    coronary_mask_dilation = torch.from_numpy(coronary_mask_np).to(device=ddf.device, dtype=torch.float32)  # (W, H, D)

    # step1 将形变后图像中的冠脉区域用边缘值替代，得到的图像作为底图
    warped_image = image_warp(cropped_image, ddf)
    assert isinstance(warped_image, torch.Tensor)
    warped_coronary = to_binary(warp_label(coronary_mask_dilation[None, None], ddf))
    assert isinstance(warped_coronary, torch.Tensor)
    warped_coronary_np = warped_coronary.squeeze().cpu().numpy()
    dte = distance_transform_edt(warped_coronary_np, return_indices=True)
    assert isinstance(dte, tuple) and len(dte) == 2
    _, (ix, iy, iz) = dte
    warped_image_at_coronary_margin = warped_image[:, :, ix, iy, iz]
    warped_image[warped_coronary==1] = warped_image_at_coronary_margin[warped_coronary==1]

    # step2 将密集速度场中coronary区域形变替换为最近的中心线的速度，得到新的密集速度场，并使用dvf2ddf得到新的密集形变场
    centerline = skeletonize(cropped_coronary_mask)
    dte = distance_transform_edt(~centerline, return_indices=True)
    assert isinstance(dte, tuple) and len(dte) == 2
    _, (ix, iy, iz) = dte
    dvf_at_nearest_centerline = dvf[:, :, ix, iy, iz]
    dvf[:, :, coronary_mask_dilation==1] = dvf_at_nearest_centerline[:, :, coronary_mask_dilation==1]
    # dvf[:, :, coronary_mask_dilation==0] = 0
    ddf = dvf2ddf(dvf)
    assert isinstance(ddf, torch.Tensor)

    # step3 将形变场应用到图像和冠脉label上
    warped_image_new = image_warp(cropped_image, ddf)
    warped_coronary = to_binary(warp_label(torch.from_numpy(cropped_coronary_mask.copy()).to(device=ddf.device, dtype=torch.float32)[None, None], ddf))
    warped_coronary_dilation = to_binary(warp_label(coronary_mask_dilation[None, None], ddf))

    # step4 将最后形变得到的图像的冠脉区域提取出来，并替换到step1得到的底图上
    warped_image[warped_coronary_dilation==1] = warped_image_new[warped_coronary_dilation==1]
    assert isinstance(warped_coronary, torch.Tensor)
    return (
        warped_image.squeeze().cpu().numpy(),
        warped_coronary.squeeze().cpu().numpy().astype(np.uint8),
        ddf.squeeze().cpu().numpy()
    )