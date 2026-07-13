from __future__ import annotations

from typing import TypeVar
from enum import StrEnum

class CoronaryType(StrEnum):
    LCA = "LCA"
    RCA = "RCA"


MM = float | int
Pixel = int
MMPerPixel = float | int
Degree = float | int
DegreePerSec = float | int
Radian = float | int

Angle = TypeVar("Angle", Degree, Radian)
type Rot[Angle] = tuple[Angle, Angle, Angle]

Sec = float | int

Point: tuple[float, float, float]