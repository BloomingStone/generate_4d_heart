from dataclasses import dataclass

import pyvista as pv


@dataclass
class Artifact:
    """A mesh-based artificial structure (guide wire, marker, stent, etc.)
    that can be added to DSA simulation."""

    mesh: pv.PolyData
    """The mesh representation of the artifact in world coordinates."""

    mu: float
    """The linear attenuation coefficient of the artifact (mm^-1)."""

    name: str = ""
    """Optional identifier for the artifact."""