from enum import StrEnum
from typing import TypeVar, Generic

import torch
import numpy as np

from . rotate_dsa.types import CoronaryType

class CoordinateSystem(StrEnum):
    WORLD = "world"
    LCA_CENTERING = "lca_centering"
    RCA_CENTERING = "rca_centering"
    
    def to_cor_type(self) -> CoronaryType | None:
        if self == CoordinateSystem.LCA_CENTERING:
            return CoronaryType.LCA
        elif self == CoordinateSystem.RCA_CENTERING:
            return CoronaryType.RCA
        else:
            return None
    
    @staticmethod
    def from_cor_type(cor_type: CoronaryType) -> "CoordinateSystem":
        if cor_type == CoronaryType.LCA:
            return CoordinateSystem.LCA_CENTERING
        elif cor_type == CoronaryType.RCA:
            return CoordinateSystem.RCA_CENTERING
        else:
            raise ValueError(f"Invalid coronary type: {cor_type}")


CS = TypeVar("CS", bound=CoordinateSystem)

class MedImage(
    Generic[CS],
):
    data: torch.Tensor
    affine: np.ndarray
    coord: CS