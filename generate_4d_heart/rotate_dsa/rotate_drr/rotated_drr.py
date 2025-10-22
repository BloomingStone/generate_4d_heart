from dataclasses import dataclass, asdict
from math import radians
from typing import Protocol, TypeVar

import torch
import numpy as np

from ..types import *
from ..cardiac_phase import CardiacPhase


@dataclass
class CArmGeometry:
    sdd: MM = 1800             # Source to detector distance
    sod: MM = 1400             # Source to object distance
    height: Pixel = 512           # Height of image
    _width: Pixel | None = None    # Width of image, default to height
    delx: MMPerPixel = 0.3           # Pixel size in x direction
    _dely: MMPerPixel | None = None   # Pixel size in y direction, default to delx
    x0: MM = 0.0             # detector principal point x-offset
    y0: MM = 0.0             # detector principal point y-offset
    
    def __post_init__(self):
        if self._width is None:
            self._width = self.height
        if self._dely is None:
            self._dely = self.delx
    
    @property
    def width(self) -> Pixel:
        assert self._width is not None
        return self._width
    
    @property
    def dely(self) -> MMPerPixel:
        assert self._dely is not None
        return self._dely

    def to_dict(self) -> dict:
        res = asdict(self)
        res["width"] = res.pop("_width")
        res["dely"] = res.pop("_dely")
        return res

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


class RotateDRR(Protocol):
    c_arm_cfg: CArmGeometry
    rotate_cfg: RotatedParameters
    
    label_center_voxel: tuple[int, int, int] | None
    
    def get_projection_at_frame(
        self, 
        frame: int,
        volume: torch.Tensor,
        coronary: torch.Tensor,
        affine: np.ndarray
    ) -> torch.Tensor:
        """
        Get drr image at given frame
        Args:
            frame (int): frame number, starting from 0
        Returns:
            torch.Tensor: shape = (c, h, w), c = 1, h = c_arm_geometry.height, w = c_arm_geometry.width
        """
        ...
    
    def get_R_T_at_frame(
        self,
        frame: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get Rotation(R) and Translation(T) of source (or camera). It may be variant for different drr projectioners.
        R and T can be used as calculate world to camera matrix.

        Args:
            frame (int): frame_index

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Rotation(R, shape=(3, 3)) and Translation(T, shape=(3,))
        """
        ...
    
    def get_additional_config(self) -> dict:
        """
        Get additional config for drr besides `c_arm_geometry`, `rotate_parameters` and `label_center_voxel` for output in json file.
        """
        return {}
    
    @property
    def image_size(self) -> tuple[Pixel, Pixel]:
        return (self.c_arm_cfg.height, self.c_arm_cfg.width)
