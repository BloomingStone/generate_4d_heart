from pathlib import Path

from monai.networks.nets.voxelmorph import VoxelMorphUNet
from monai.networks.blocks.warp import DVF2DDF, Warp
from monai.transforms import (EnsureChannelFirst, AsDiscrete, ToTensor, Compose) # type: ignore
import torch
from torch import nn
import numpy as np
from nibabel.nifti1 import Nifti1Image


from .. import NUM_TOTAL_CAVITY_LABEL
from .utils import MaybeFlipTransform

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

class ShapeMorphPredictor:
    def __init__(
            self, 
            checkpoint_path: Path, 
            device_id: int = 0,
        ):
        self.device = torch.device(device_id)
        self.model = ShapeMorph(label_class_num=NUM_TOTAL_CAVITY_LABEL, norm='SyncBatch').to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

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
            source_cavity_zoomed: Nifti1Image
        ):
        self.source_cavity_zoomed = source_cavity_zoomed

        assert source_cavity_zoomed.affine is not None
        self.flips = MaybeFlipTransform(source_cavity_zoomed.affine)
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
    ) -> Nifti1Image:
        """predict the dvf from source cavity to target cavity

        Args:
            target_cavity_zoomed (Nifti1Image): target cavity zoomed image (shape = 144x144x128)

        Returns:
            Nifti1Image: dvf from source cavity to target cavity (shape = 144x144x128x1x3)
        """
        assert (
            self.flips is not None and
            self.source_cavity_zoomed is not None
        )

        target_cavity_tensor = self._get_tensor(target_cavity_zoomed, self.transform_cavity)

        with torch.no_grad():
            dvf: torch.Tensor = self.model(self.source_cavity_tensor, target_cavity_tensor, return_dvf=True)

        dvf = self.flips(dvf, is_vector_field=True)   # 当对位移场进行翻转时需要注意，可能需要将内部的值一并进行翻转
        dvf_np = dvf.squeeze().detach().cpu().numpy()
        dvf_np = np.transpose(dvf_np, (1, 2, 3, 0))[:, :, :, None, :]  # shape = (W, H, D, 1, 3)
        dvf_nii = Nifti1Image(dvf_np, self.source_cavity_zoomed.affine)

        return dvf_nii