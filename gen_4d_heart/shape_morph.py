from pathlib import Path

from monai.networks.nets.voxelmorph import VoxelMorphUNet
from monai.networks.blocks.warp import DVF2DDF, Warp
from monai.transforms import (EnsureChannelFirst, AsDiscrete, ToTensor, Compose, Transform) # type: ignore
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from nibabel.nifti1 import Nifti1Image
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_dilation

from . import SSM_SHAPE, NUM_TOTAL_CAVITY_LABEL, SSM_DIRECTION
from .roi import ROI
from .utils import get_maybe_flip_transform

class ShapeMorph(nn.Module):
    def __init__(
            self,
            *,
            label_class_num: int,
            integration_steps: int = 7,
            backbone: VoxelMorphUNet | nn.Module | None = None,
            norm: tuple | str | None = None,
    ):
        super().__init__()

        in_channels = (label_class_num + 1)*2
        
        self.stem = nn.Conv3d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0
        )

        if backbone is None:
            self.backbone = VoxelMorphUNet(
                spatial_dims=3,
                in_channels=16,
                unet_out_channels=32,
                channels=(16, 32, 32, 32, 32, 32),
                final_conv_channels=(16, 16),
                norm=norm
            )
        
        self.integration_steps = integration_steps
        self.diffeomorphic = True if self.integration_steps > 0 else False
        if self.diffeomorphic:
            self.dvf2ddf = DVF2DDF(num_steps=self.integration_steps, mode="bilinear", padding_mode="zeros")
        self.warp = Warp(mode="bilinear", padding_mode="zeros")
    
    def forward(
            self, 
            original_label_onehot: torch.Tensor,
            target_label_onehot: torch.Tensor,
            return_dvf: bool = False
    ) -> torch.Tensor:
        x = self.stem(torch.cat([original_label_onehot, target_label_onehot], dim=-4))
        x = self.backbone(x)
        if self.diffeomorphic and not return_dvf:
            x = self.dvf2ddf(x)
        
        return x

def get_coronary_centerline_ddf(
        model: ShapeMorph,
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
    image_warp = model.warp
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

class ShapeMorphPredictor:
    def __init__(
            self, 
            checkpoint_path: Path, 
            device_id: int = 0,
            return_wrapped_image: bool = False,
            return_wrapped_coronary: bool = False
        ):
        self.device = torch.device(device_id)
        self.model = ShapeMorph(label_class_num=NUM_TOTAL_CAVITY_LABEL, norm='SyncBatch').to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.return_wrapped_image = return_wrapped_image
        self.return_wrapped_coronary = return_wrapped_coronary

        # persistent data
        self.source_cavity_zoomed = None
        self.coronary = None
        self.original_image = None
        self.roi = None
        self.source_cavity_tensor = None
        self.transform_cavity = None
        self.flips = None

    def set_shared_inputs(
            self, 
            source_cavity_zoomed: Nifti1Image,
            coronary: Nifti1Image,
            original_image: Nifti1Image,
            roi: ROI
        ):
        self.source_cavity_zoomed = source_cavity_zoomed
        self.coronary = coronary
        self.original_image = original_image
        self.roi = roi

        assert source_cavity_zoomed.affine is not None
        self.flips = get_maybe_flip_transform(source_cavity_zoomed.affine)
        self.transform_cavity = Compose([
            ToTensor(),
            self.flips,
            EnsureChannelFirst(channel_dim='no_channel'),
            AsDiscrete(to_onehot=NUM_TOTAL_CAVITY_LABEL + 1)
        ])
        self.source_cavity_tensor = self._get_tensor(source_cavity_zoomed, self.transform_cavity)

    def _get_tensor(self, image: Nifti1Image, transform) -> torch.Tensor:
        image_np = image.get_fdata().astype(np.float16)
        image_tensor = transform(image_np)
        assert isinstance(image_tensor, torch.Tensor)
        image_tensor = image_tensor[None].to(device=self.device, dtype=torch.float32)
        assert image_tensor.dim() == 5
        return image_tensor

    def predict(
            self, 
            target_cavity_zoomed: Nifti1Image
    ) -> dict[str, Nifti1Image]:
        assert (
            self.roi is not None and
            self.flips is not None and
            self.original_image is not None and
            self.coronary is not None and
            self.source_cavity_zoomed is not None
        )

        target_cavity_tensor = self._get_tensor(target_cavity_zoomed, self.transform_cavity)

        with torch.no_grad():
            dvf: torch.Tensor = self.model(self.source_cavity_tensor, target_cavity_tensor, return_dvf=True)
            zoom_rate = 1 / self.roi.get_zoom_rate().reshape(1, 3, 1, 1, 1)
            dvf = F.interpolate(dvf, scale_factor=zoom_rate.flatten().tolist(), mode="trilinear", align_corners=False)
            dvf = dvf * torch.from_numpy(zoom_rate).to(self.device, dtype=torch.float32)

        # warp image and coronary
        transform_image = Compose([ToTensor(), self.flips, EnsureChannelFirst(channel_dim='no_channel')])
        cropped_image_tensor = self._get_tensor(self.roi.crop(self.original_image), transform_image)
        cropped_coronary_np = self.flips(self.roi.crop(self.coronary).get_fdata().astype(np.int8))

        with torch.no_grad():
            warped_image, warped_coronary, ddf = get_coronary_centerline_ddf(self.model, dvf, cropped_image_tensor, cropped_coronary_np)

        ddf = self.flips(ddf)
        ddf = np.transpose(ddf, (1, 2, 3, 0))[:, :, :, None, :]  # shape = (W, H, D, 1, 3)
        ddf_nii = Nifti1Image(ddf, self.source_cavity_zoomed.affine)

        res = {'ddf': ddf_nii}

        if self.return_wrapped_image:
            wrapped_image = self.roi.recover_cropped(self.flips(warped_image), background=self.original_image)
            res['image'] = wrapped_image

        if self.return_wrapped_coronary:
            wrapped_coronary = self.roi.recover_cropped(self.flips(warped_coronary))
            res['coronary'] = wrapped_coronary

        return res