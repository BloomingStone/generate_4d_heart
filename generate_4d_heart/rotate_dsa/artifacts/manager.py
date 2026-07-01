from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

from generate_4d_heart import MU_GUIDE_WIRE, MU_MARKER
from .artifact import Artifact
from .guide_wire import generate_guide_wire
from .marker import generate_markers


# ---- Configuration dataclasses ------------------------------------------------

@dataclass
class GuideWireConfig:
    radius_mm: float = 0.2
    mu: float = MU_GUIDE_WIRE


@dataclass
class MarkerConfig:
    distance_mm: float = 6.0
    radius_mm: float = 0.5
    mu: float = MU_MARKER
    margin_ratio: float = 0.9


@dataclass
class ExternalArtifactConfig:
    path: Path
    mu: float
    name: str = ""


@dataclass
class ArtifactManagerConfig:
    guide_wire: GuideWireConfig | None = None
    markers: MarkerConfig | None = None
    external_files: list[ExternalArtifactConfig] = field(default_factory=list)


# ---- ArtifactManager ---------------------------------------------------------

class ArtifactManager:
    """Config-driven factory for generating artifacts from coronary labels.

    Handles artifact dependency order:
    1. Guide wire (requires coronary + cavity labels)
    2. Markers (requires guide wire centerline)
    3. External mesh files (loaded directly)
    """

    def __init__(self, config: ArtifactManagerConfig):
        self.config = config

    def generate(
        self,
        coronary_label: np.ndarray,
        cavity_label: np.ndarray,
        affine: np.ndarray,
    ) -> list[Artifact]:
        """Generate all configured artifacts.

        Parameters
        ----------
        coronary_label : np.ndarray
            3D binary/uint8 mask of a single coronary artery.
        cavity_label : np.ndarray
            3D uint8 cavity label map.
        affine : np.ndarray
            4x4 affine matrix (voxel -> world).

        Returns
        -------
        list[Artifact]
            All generated artifacts.
        """
        artifacts: list[Artifact] = []
        centerline_world: np.ndarray | None = None

        # 1. Guide wire
        if self.config.guide_wire is not None:
            gw_cfg = self.config.guide_wire
            gw_artifact, centerline_world = generate_guide_wire(
                coronary_label=coronary_label,
                cavity_label=cavity_label,
                affine=affine,
                wire_radius_mm=gw_cfg.radius_mm,
                mu=gw_cfg.mu,
            )
            artifacts.append(gw_artifact)

        # 2. Markers (require guide wire centerline)
        if self.config.markers is not None:
            if centerline_world is None:
                raise ValueError(
                    "MarkerConfig requires a guide wire to be generated first. "
                    "Set guide_wire in ArtifactManagerConfig."
                )
            mk_cfg = self.config.markers
            marker_artifacts = generate_markers(
                centerline_world=centerline_world,
                marker_distance_mm=mk_cfg.distance_mm,
                marker_radius_mm=mk_cfg.radius_mm,
                mu=mk_cfg.mu,
                margin_ratio=mk_cfg.margin_ratio,
            )
            artifacts.extend(marker_artifacts)

        # 3. External mesh files
        for ext_cfg in self.config.external_files:
            if not ext_cfg.path.exists():
                raise FileNotFoundError(f"External artifact mesh not found: {ext_cfg.path}")
            mesh = pv.read(ext_cfg.path)
            name = ext_cfg.name if ext_cfg.name else ext_cfg.path.stem
            artifacts.append(Artifact(mesh=mesh, mu=ext_cfg.mu, name=name))

        return artifacts
