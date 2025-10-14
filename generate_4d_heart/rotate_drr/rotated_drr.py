from dataclasses import dataclass
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


@dataclass
class RotatedParameters:
    total_frame: int = 120
    alpha_start: Degree = 30.0      # Primary rotation angle
    beta_start: Degree = 0.0        # Secondary rotation angle
    angular_velocity: DegreePerSec = 75.0 # Angular velocity of alpha
    fps: float = 60.0                 # Frame per second of DSA
    
    def get_angle_at_frame(self, frame: int) -> Rot[Degree]:
        d_alpha = frame * self.angular_velocity / self.fps
        new_alpha = self.alpha_start - d_alpha
        return (new_alpha, self.beta_start, 0.0)
    
    def get_angle_at_frame_radian(self, frame: int) -> Rot[Radian]:
        a, b, c = self.get_angle_at_frame(frame)
        return (radians(a), radians(b), radians(c))


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
        
    @property
    def image_size(self) -> tuple[Pixel, Pixel]:
        return (self.c_arm_cfg.height, self.c_arm_cfg.width)
