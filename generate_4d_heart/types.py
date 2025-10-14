from typing import TypeVar

MM = float | int
Pixel = int
MMPerPixel = float | int
Degree = float | int
DegreePerSec = float | int
Radian = float | int

Angle = TypeVar("Angle", Degree, Radian)
type Rot[Angle] = tuple[Angle, Angle, Angle]

Sec = float | int