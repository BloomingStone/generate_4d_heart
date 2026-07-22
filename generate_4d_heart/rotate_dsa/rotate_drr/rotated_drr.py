from dataclasses import dataclass, asdict
from math import radians
from typing import Protocol, Literal

import torch
import numpy as np

from ...types import *
from ...med_image import MedImage, Centering


@dataclass
class CArmGeometry:
    sdd: MM = 1200
    """Source to detector distance (mm)"""
    
    sod: MM = 760
    """Source to object distance (mm)"""
    
    height: Pixel = 512
    """Height of image"""
    
    width: Pixel = 512
    """Width of image"""
    
    delx: MMPerPixel = 0.368
    """Pixel size in x direction (mm/pixel)"""
    
    dely: MMPerPixel = 0.368
    """Pixel size in y direction (mm/pixel)"""
    
    x0: MM = 0.0
    """Detector principal point x-offset (mm)"""
    
    y0: MM = 0.0
    """Detector principal point y-offset (mm)"""
    
    
    orientation_type: Literal["AP", "PA"] = "PA"
    """
    Orientation type of the C-arm. 
    "AP" means Anterior-Posterior, where the X-ray source is initially in front of the patient, 
    and the detector is behind the patient, so the camera is looking from front to back. 
    "PA" means Posterior-Anterior, where the X-ray source is behind the patient. 
    "PA" is the default orientation as in DSA C-Arm.
    """

    def to_dict(self) -> dict:
        return asdict(self)
    

@dataclass
class RotatedParameters:
    """
    Parameters for rotated DRR.
    
    By default, the coordianate system is RAS, which means X axis is to the right of patient, Y axis is to the anterior of patient, and Z axis is to the superior of patient.
    
    The rotation type is "euler_angles" and the order is ZXY, alpha is primary rotation angle, beta is secondary rotation angle, so the rotation first rotate around Z axis(SI axis) by alpha, then around X axis (RL axis) by beta, and finally around Y axis(AP axis).
    
    For now, only alpha will change, with alpha_f = alpha_start - d_alpha, d_alpha = frame * angular_velocity / fps
    """
    total_frame: int = 90                   # Total frame of DSA; 180 for real sence, 90 for simulation, totally 3 seconds
    alpha_start: Degree = 30.0              # Primary rotation angle
    beta_start: Degree = 0.0                # Secondary rotation angle
    angular_velocity: DegreePerSec = 75.0   # Angular velocity of alpha
    fps: float = 30.0                       # Frame per second of DSA; 60 fps for real sence, 30 fps for simulation
    coordinate_system: str = "RAS"          # X is R, Y is A, Z is S
    parameterization: str = "euler_angles"  # representation of rotation
    convention: str = "ZXY"                 # rotation axis sequence, internal rotation
    
    def get_rotation_angle_at_frame(self, frame: int) -> Rot[Degree]:
        d_alpha = frame * self.angular_velocity / self.fps
        new_alpha = self.alpha_start + d_alpha
        return (new_alpha, self.beta_start, 0.0)
    
    def get_rotaiton_radian_at_frame(self, frame: int) -> Rot[Radian]:
        a, b, c = self.get_rotation_angle_at_frame(frame)
        return (radians(a), radians(b), radians(c))

    def to_dict(self) -> dict:
        return asdict(self)
    
    @staticmethod
    def from_alpha_range(
        total_frame: int,
        alpha_start: Degree,
        alpha_end: Degree,
        fps: float = 30.0,
        beta_start: Degree = 0.0,
        coordinate_system: str = "RAS",
        parameterization: str = "euler_angles",
        convention: str = "ZXY"
    ) -> RotatedParameters:
        angular_velocity = (alpha_end - alpha_start) * fps / total_frame
        return RotatedParameters(
            total_frame=total_frame,
            alpha_start=alpha_start,
            beta_start=beta_start,
            angular_velocity=angular_velocity,
            fps=fps,
            coordinate_system=coordinate_system,
            parameterization=parameterization,
            convention=convention
    )


class RotateDRR(Protocol):
    c_arm_cfg: CArmGeometry
    rotate_cfg: RotatedParameters

    def get_projection_at_frame(
        self,
        frame: int,
        volume: MedImage[Centering],
        coronary: MedImage[Centering],
    ) -> torch.Tensor:
        """
        Get DRR image at given frame.
        volume 和 coronary 必须是同一坐标系（通常为 Centering），共享相同 affine。
        """
        ...

    def get_projections_at_frames(
        self,
        frames: list[int],
        volume: MedImage[Centering],
        coronary: MedImage[Centering],
    ) -> torch.Tensor:
        """
        Get DRR images at given frames (batch render).
        """
        ...

    def get_R_T_at_frame(
        self,
        frame: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get Rotation(R) and Translation(T) of source (or camera). It may be variant for different DRR projectioners.
        R and T can be used as calculate world to camera matrix.
        usually z is the direction of X-rays are emitted

        Args:
            frame (int): frame_index

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Rotation(R, shape=(3, 3)) and Translation(T, shape=(3,))
        """
        ...
    
    def get_additional_config(self) -> dict:
        """
        Get additional config for DRR besides `c_arm_geometry`, `rotate_parameters` for output in json file.
        """
        return {}
    
    @property
    def image_size(self) -> tuple[Pixel, Pixel]:
        return (self.c_arm_cfg.height, self.c_arm_cfg.width)
    
    @property
    def device(self) -> torch.device:
        ...
