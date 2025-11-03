from typing import TypeVar
from enum import StrEnum

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

class CoronaryType(StrEnum):
    LCA = "LCA"
    RCA = "RCA"