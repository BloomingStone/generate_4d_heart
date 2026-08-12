from typing import Literal, Optional, Sequence
from math import radians

import numpy as np
import torch
from torchio import LabelMap, ScalarImage, Subject
from diffdrr.drr import DRR
from diffdrr.pose import convert, RigidTransform

from .rotated_drr import RotateDRR, CArmGeometry, RotatedParameters
from ...med_image import MedImage, Centering
from ...types import Degree
from ... import MU_WATER


def get_reorientation(
        orientation_type: Optional[Literal["AP", "PA"]] = "AP"
) -> torch.Tensor:
    """
    Get the reorientation matrix for different X-ray orientations.
    
    ***** ATTNTION: "CAMERA/SOURCE" COORDIANATE SYSTEM *****
    DiffDRR's "camera" uses a coordinate system where:
    - The Z-axis points forward (towards the X-ray source).
    - The Y-axis points upward.
    - The X-axis points to the left.
    which is different from OpenGL or COLMAP.  Similar to Pytorch3D
    
    Args:
        orientation_type: The desired orientation type ("AP" or "PA").
            "AP" means Anterior-Posterior, where the X-ray source is initially in front of the patient, and the detector is behind the patient, so the camera is looking from front to back.
            "PA" means Posterior-Anterior, where the X-ray source is behind the patient, and the detector is in front of the patient, so the camera is looking from back to front.
    Returns:
        A 4x4 reorientation matrix as a torch tensor.
    """
    
    if orientation_type == "AP":
        # Initially, the source is in front of the patient, and the detector is behind the patient, so the camera is looking from front to back.
        # Rotates 90 degrees around the x-axis
        reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    elif orientation_type == "PA":
        # Initially, the source is behind the patient, and the detector is in front of the patient, so the camera is looking from back to front.
        # Rotates 90 degrees around the x-axis and then 180 degrees around the y-axis
        reorient = torch.tensor(
            [
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    elif orientation_type is None:
        # Identity transform
        reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unrecognized orientation {orientation_type}")
    return reorient


class TorchDRR(RotateDRR):
    def __init__(
        self, 
        c_arm_cfg: CArmGeometry = CArmGeometry(),
        rotate_cfg: RotatedParameters = RotatedParameters(),
        patch_size: int = 256,
    ):
        self.c_arm_cfg = c_arm_cfg
        self.rotate_cfg = rotate_cfg
        self.patch_size = patch_size
        self.diff_drr: DRR | None = None
        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            raise RuntimeError("No CUDA device available for TorchDRR")
        self.reorient = get_reorientation(c_arm_cfg.orientation_type)
        sod = self.c_arm_cfg.sod
        if c_arm_cfg.orientation_type == "AP":
            self.translations = torch.tensor([[0.0, sod, 0.0]], device=self.device)
        elif c_arm_cfg.orientation_type == "PA":
            # PA, source is behind the patient, so the translation is negative along the Y-axis
            self.translations = torch.tensor([[0.0, -sod, 0.0]], device=self.device)
        else:
            raise ValueError(f"orientation_type must be either AP or PA, but got {c_arm_cfg.orientation_type}")


    def _setup(
        self,
        volume: MedImage[Centering],
        coronary: MedImage[Centering],
    ):
        vol = volume.data
        mask = coronary.data
        aff = volume.affine
        assert vol.dim() >= 3
        w, h, d = vol.shape[-3:]
        shape = (1, w, h, d)

        subject = Subject(
            volume=ScalarImage(tensor=vol.float().reshape(*shape).to(self.device), affine=aff),
            mask=LabelMap(tensor=mask.reshape(*shape).to(self.device), affine=aff),
            reorient=self.reorient,  # type: ignore
            density=ScalarImage(tensor=vol.float().reshape(*shape).to(self.device), affine=aff),
            fiducials=None,  # type: ignore
        )
        
        geo = self.c_arm_cfg
        self.diff_drr = DRR(
            subject=subject,
            sdd = geo.sdd,
            height=geo.height,
            width=geo.width,
            delx=geo.delx,
            dely=geo.dely,
            x0=geo.x0,
            y0=geo.y0,
            patch_size=self.patch_size,
            renderer = "trilinear",
            compile_renderer=True,
            checkpoint_gradients=True,
        ).to(self.device).eval()
        
    
    
    def _get_projection_after_setup(
        self,
        rotations: torch.Tensor,
    ):
        assert self.diff_drr is not None, "diff_drr is None, call setup() first"
        drr = self.diff_drr
        N, D = rotations.shape
        assert D == 3
        
        def do_drr(rots: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
            # [Positioner Primary Angle Attribute – DICOM Standard Browser](https://dicom.innolitics.com/ciods/x-ray-angiographic-image/xa-positioner/00181510)
            res = drr(
                rots.to(self.device),     # here has negtived
                trans.to(self.device), 
                parameterization=self.rotate_cfg.parameterization, 
                convention=self.rotate_cfg.convention,
            )
            return res
        
        if N == 1:
            # in DiffDrr, X Y Z coreesponding to R A S, the rotate angle is oppsite to standard C-Arm 
            # X-ray, where X Y Z cooresponding to I L A
            # the convert of R, T, angles and camera to world matrix can be found at test/test_R_T.py
            return do_drr(rotations, self.translations)
        
        translations = self.translations.repeat(N, 1)
        # TODO auto calculate the max batch size at limits of CUDA memory.
        # For now, use 1 batch
        res = []
        for rot, trans in zip(rotations, translations):
            drr_img = do_drr(rot.unsqueeze(0), trans.unsqueeze(0))
            res.append(drr_img)
        drr_img = torch.cat(res, dim=0)
        return drr_img
    
    
    def get_projections_at_frames(
        self,
        frames: list[int],
        volume: MedImage[Centering],
        coronary: MedImage[Centering],
    ) -> torch.Tensor:
        """
        Batch-render DRR at multiple frames with a single _setup().
        Returns shape (N, H, W), line integrals.
        """
        self._setup(volume, coronary)
        rotations = torch.tensor(
            [self.rotate_cfg.get_rotaiton_radian_at_frame(f) for f in frames],
            device=self.device,
        )
        return self._get_projection_after_setup(rotations)

    def get_projection_at_frame(
        self,
        frame: int,
        volume: MedImage[Centering],
        coronary: MedImage[Centering],
    ) -> torch.Tensor:
        """return shape=(T, C, W, H), C=0: image, C=1: coronary_mask, C=2: depthmask"""
        self._setup(volume, coronary)

        rotation = self.rotate_cfg.get_rotaiton_radian_at_frame(frame)
        rotations = torch.tensor([rotation], device=self.device)

        return self._get_projection_after_setup(rotations)
    
    def get_R_T_at_frame(self, frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        # in DiffDrr, X Y Z coreesponding to R A S, the rotate angle is oppsite to standard C-Arm 
        # X-ray, where X Y Z cooresponding to I L A
        # the convert of R, T, angles and camera to world matrix can be found at test/test_R_T.py
        rotation = self.rotate_cfg.get_rotaiton_radian_at_frame(frame)
        rotations = torch.tensor([rotation], device=self.device)
        # [Positioner Primary Angle Attribute – DICOM Standard Browser](https://dicom.innolitics.com/ciods/x-ray-angiographic-image/xa-positioner/00181510)
        rot = convert(
            rotations.cpu(), # here has negtived
            self.translations.cpu(), 
            parameterization=self.rotate_cfg.parameterization, 
            convention=self.rotate_cfg.convention
        )
        reorient = RigidTransform(self.reorient)
        pose = reorient.compose(rot)    # Rot @ reorient
        
        R = pose.rotation
        T = pose.translation
        
        return (R, T)
    
    def get_projections_at_degrees(
        self,
        angles: Degree | Sequence[Degree],
        volume: MedImage[Centering],
        coronary: MedImage[Centering],
    ) -> torch.Tensor:
        if isinstance(angles, float | int):
            angles = [angles]
        rotations = torch.tensor([[radians(a), 0.0, 0.0] for a in angles], device=self.device)
        self._setup(volume, coronary)

        assert rotations.dim() == 2 and rotations.shape[1] == 3
        assert rotations.dtype == torch.float32
        rotations = rotations.to(self.device)

        return self._get_projection_after_setup(rotations)
    
    
    def get_additional_config(self) -> dict:
        return {
            "patch_size": self.patch_size,
        }
    
    @property
    def device(self) -> torch.device:
        return self._device