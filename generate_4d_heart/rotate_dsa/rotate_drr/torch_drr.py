from typing import Literal, Optional, Sequence
from math import radians

import numpy as np
from scipy.ndimage import center_of_mass
import torch
from torchio import LabelMap, ScalarImage, Subject
from diffdrr.drr import DRR
from diffdrr.pose import convert, RigidTransform

from .rotated_drr import RotateDRR, CArmGeometry, RotatedParameters
from ..types import Degree


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

def recenter(
    original_affine: np.ndarray,
    center_voxel: tuple[int, int, int]
) -> np.ndarray:
    """set the center of the image to the given center_voxel
    """
    B = original_affine[:3, :3]
    new_t = -(B @ np.array(center_voxel))
    T = np.eye(4)
    T[:3, :3] = B
    T[:3, 3] = new_t
    return T


class TorchDRR(RotateDRR):
    def __init__(
        self, 
        c_arm_cfg: CArmGeometry = CArmGeometry(),
        rotate_cfg: RotatedParameters = RotatedParameters(),
        patch_size: int = 256,
        orientation_type: Optional[Literal["AP", "PA"]] = "AP"
    ):
        self.c_arm_cfg = c_arm_cfg
        self.rotate_cfg = rotate_cfg
        self.patch_size = patch_size
        self.orientation_type = orientation_type
        self.diff_drr: DRR | None = None
        self.label_center_voxel: tuple[int, int, int] | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reorient = get_reorientation(self.orientation_type)
        sod = self.c_arm_cfg.sod
        self.translations = torch.tensor([[0.0, sod, 0.0]], device=self.device)
        
        
    def _setup_label_center_voxel(self, coronary: torch.Tensor):
        label_center: tuple[int, int, int] = center_of_mass(coronary.squeeze().cpu().numpy()) # type: ignore
        W, H, D = coronary.shape[-3:]
        image_center = (W/2, H/2, D/2)
        self.label_center_voxel = (
            int( (image_center[0] + label_center[0]) / 2 ), # left and right, set as the mean of image_center and label_center
            int( (image_center[1] + label_center[1]) / 2 ), # antero-posterior, same as above
            int( image_center[2] )                          # up and down, set as the center of image
        )


    def _setup(
        self,
        volume: torch.Tensor,
        coronary: torch.Tensor,
        affine: np.ndarray,
    ):
        assert volume.dim() >= 3
        w, h, d = volume.shape[-3:]
        shape = (1, w, h, d)
        
        assert self.label_center_voxel is not None
        affine = recenter(affine, self.label_center_voxel)
        
        subject = Subject(
            volume=ScalarImage(tensor=volume.reshape(*shape).to(self.device), affine=affine),
            mask=LabelMap(tensor=coronary.reshape(*shape).to(self.device),affine=affine),
            reorient = self.reorient,    # type: ignore
            density = ScalarImage(tensor=volume.reshape(*shape).to(self.device),affine=affine),
            fiducials = None,   #type: ignore
        )
        
        geo = self.c_arm_cfg
        self.diff_drr = DRR(
            subject=subject,
            sdd = geo.sdd,
            height=geo.height,
            width=geo._width,
            delx=geo.delx,
            dely=geo._dely,
            x0=geo.x0,
            y0=geo.y0,
            patch_size=self.patch_size,
        ).to(self.device)
    
    
    def _get_projection_after_setup(
        self,
        rotations: torch.Tensor
    ):
        assert self.diff_drr is not None, "diff_drr is None, call setup() first"
        N, D = rotations.shape
        assert D == 3
        if N == 1:
            return self.diff_drr(
                rotations.to(self.device), 
                self.translations.to(self.device), 
                parameterization=self.rotate_cfg.parameterization, 
                convention=self.rotate_cfg.convention
            )
        
        
        translations = self.translations.repeat(N, 1)
        # TODO auto calculate the max batch size at limits of CUDA memory.
        # For now, use 1 batch
        res = []
        for rot, trans in zip(rotations, translations):
            drr_img = self.diff_drr(
                rot.unsqueeze(0).to(self.device), 
                trans.unsqueeze(0).to(self.device), 
                parameterization=self.rotate_cfg.parameterization, 
                convention=self.rotate_cfg.convention
            )
            res.append(drr_img)
        drr_img = torch.cat(res, dim=0)
        return drr_img
    
    
    def get_projection_at_frame(
        self, 
        frame: int,
        volume: torch.Tensor,
        coronary: torch.Tensor,
        affine: np.ndarray,
    ) -> torch.Tensor:
        if frame == 0:
            self._setup_label_center_voxel(coronary)
        
        self._setup(volume, coronary, affine)
        
        rotation = self.rotate_cfg.get_rotaiton_radian_at_frame(frame)
        rotations = torch.tensor([rotation], device=self.device)
        
        return self._get_projection_after_setup(rotations)
    
    def get_R_T_at_frame(self, frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        rotation = self.rotate_cfg.get_rotaiton_radian_at_frame(frame)
        rotations = torch.tensor([rotation], device=self.device)
        rot = convert(
            rotations.cpu(), 
            self.translations.cpu(), 
            parameterization=self.rotate_cfg.parameterization, 
            convention=self.rotate_cfg.convention
        )
        reorient = RigidTransform(self.reorient)
        pose = reorient.compose(rot)
        
        R = pose.rotation
        T = pose.translation
        
        return (R, T)
    
    def get_projections_at_degrees(
        self, 
        angles: Degree | Sequence[Degree],
        volume: torch.Tensor,
        coronary: torch.Tensor,
        affine: np.ndarray,
        make_center_at_coronary: bool = True
    ) -> torch.Tensor:
        if isinstance(angles, float | int):
            angles = [angles]
        rotations = torch.tensor([[radians(a), 0.0, 0.0] for a in angles], device=self.device)
        
        if make_center_at_coronary:
            self._setup_label_center_voxel(coronary)
        self._setup(volume, coronary, affine)
        
        assert rotations.dim() == 2 and rotations.shape[1] == 3
        assert rotations.dtype == torch.float32
        rotations = rotations.to(self.device)
        
        return self._get_projection_after_setup(rotations)
    
    
    def get_projections_at_frames(
        self,
        frames: list[int],
        volume: torch.Tensor,
        coronary: torch.Tensor,
        affine: np.ndarray,
        make_center_at_coronary: bool = True
    ) -> torch.Tensor:
        ...
        raise NotImplementedError() # TODO
