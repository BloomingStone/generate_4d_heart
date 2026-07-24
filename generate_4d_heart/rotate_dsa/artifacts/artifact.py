from dataclasses import dataclass
from typing import Final

import pyvista as pv
import numpy as np


@dataclass
class Artifact:
    """A mesh-based artificial structure (guide wire, marker, stent, etc.)
    that can be added to DSA simulation."""

    mesh: pv.PolyData
    """The mesh representation of the artifact in world coordinates."""

    mu: Final[float]
    """The linear attenuation coefficient of the artifact (mm^-1)."""

    name: Final[str] = ""
    """Optional identifier for the artifact."""
    
    def __post_init__(self):
        self.mesh.point_data["mu"] = np.full(self.mesh.n_points, self.mu, dtype=np.float32)

    def translate(self, vec: np.ndarray) -> "Artifact":
        """Return a new Artifact with the mesh translated by ``vec``."""
        new_mesh = self.mesh.copy()
        new_mesh.points = new_mesh.points + vec
        return Artifact(mesh=new_mesh, mu=self.mu, name=self.name)
    
    def copy(self) -> "Artifact":
        """Return a copy of the artifact."""
        return Artifact(mesh=self.mesh.copy(), mu=self.mu, name=self.name)