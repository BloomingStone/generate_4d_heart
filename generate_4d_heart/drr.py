from math import radians
from dataclasses import dataclass, asdict
from typing import Literal, Optional

import torch
from diffdrr.drr import DRR
from torchio import LabelMap, ScalarImage, Subject
from diffdrr.data import canonicalize
import numpy as np


def get_reorientation(
        orientation_type: Optional[Literal["AP", "PA"]] = "AP"
) -> torch.Tensor:
    # Frame-of-reference change
    if orientation_type == "AP":
        # Rotates the C-arm about the x-axis by 90 degrees
        # Rotates the C-arm about the z-axis by -90 degrees
        reorient = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif orientation_type == "PA":
        # Rotates the C-arm about the x-axis by 90 degrees
        # Rotates the C-arm about the z-axis by 90 degrees
        reorient = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif orientation_type is None:
        # Identity transform
        reorient = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    else:
        raise ValueError(f"Unrecognized orientation {orientation_type}")
    return reorient

@dataclass
class ProjectParameters:
    sdd: int = 1800
    sod: int = 1400
    height: int = 512
    width: int | None = None
    delx: float = 0.3
    dely: float | None = None
    x0: float = 0.0
    y0: float = 0.0
    patch_size: int = 256
    
    def __post_init__(self):
        if self.width is None:
            self.width = self.height
        if self.dely is None:
            self.dely = self.delx
            

def get_drr(
    image: torch.Tensor,
    label: torch.Tensor,
    affine: np.ndarray | None,
    rotations_degree: tuple[float, float, float],
    mean_hu_at_coronary: float,
    drr_parameters: ProjectParameters,
    device: torch.device,
) -> torch.Tensor:
    # Ensure tensors are on CPU for ScalarImage/LabelMap creation
    volume = ScalarImage(
        tensor=image.squeeze(0).to(device),
        affine=affine,
    )
    coronary_segmentation = LabelMap(
        tensor=label.squeeze(0).to(device),
        affine=affine,
    )
    
    # TODO 目前直接使用冠脉均值作为整条冠脉的密度可能有点太粗略了，可能整体乘以一个系数比较好
    # TODO 现在也有对应的标签，或许可以不用造影剂强度区分区域，而是直接使用标签作为mask区分区域
    lung_alpha = 0.7  # -600 <= Hu < 0
    heart_alpha = 0.5  # 0 <= Hu < 500
    bone_alpha = 1.5    # HU >= 600
    coronary_alpha = 13.0

    volume_data = volume.data.to(torch.float32)
    air = torch.where(volume_data <= -600)
    lung = torch.where((-600 < volume_data) & (volume_data <= 0))
    heart = torch.where((0 < volume_data) & (volume_data <= 600))
    bone = torch.where(volume_data > 600)
    coronary_segmentation_data = coronary_segmentation.data
    coronary = torch.where(coronary_segmentation_data > 0)

    density = torch.empty_like(volume_data)
    density[air] = volume_data[lung].min()
    density[lung] = volume_data[lung] * lung_alpha
    density[heart] = volume_data[heart] * heart_alpha
    density[bone] = volume_data[bone] * bone_alpha
    density[coronary] = mean_hu_at_coronary * coronary_alpha
    
    subject = Subject(
        volume = volume,
        mask = coronary_segmentation,
        reorient = get_reorientation("AP"),     # type: ignore
        density = ScalarImage(tensor=density, affine=volume.affine),
        fiducials = None,   # type: ignore
    )
    
    subject = canonicalize(subject)
    
    params = asdict(drr_parameters)
    sod = params.pop("sod")
    drr = DRR(
        subject=subject,
        **params,
    ).to(device)
    
    rotations_radians = [radians(rotation) for rotation in rotations_degree]
    rotations = torch.tensor([rotations_radians], device=device)
    translations = torch.tensor([[0.0, sod, 0.0]], device=device)

    img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
    return img

