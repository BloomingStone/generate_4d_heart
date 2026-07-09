from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

from generate_4d_heart import MU_GUIDE_WIRE, MU_MARKER
from .artifact import Artifact
from .guide_wire import generate_guide_wire
from .marker import generate_markers
from .stent import generate_stent


# ---- Configuration dataclasses ------------------------------------------------

@dataclass
class GuideWireConfig:
    radius_mm: float = 0.3
    mu: float = MU_GUIDE_WIRE


@dataclass
class MarkerConfig:
    distance_mm: float = 13
    radius_mm: float = 0.5
    mu: float = MU_MARKER
    margin_ratio: float = 0.8
    """
    Margin ratio for marker placement along the guide wire centerline, 
    to avoid placing markers too close to the ends of the centerline.
    """
    

@dataclass
class StentConfig:
    radius_mm: float = 1.5
    strut_radius_mm: float = 0.15
    n_wires: int = 6
    mu: float = MU_GUIDE_WIRE
    margin_from_marker_mm: float = 1
    """Margin in mm from the nearest marker to the stent ends."""

@dataclass
class ExternalArtifactConfig:
    path: Path
    mu: float
    name: str = ""


@dataclass
class ArtifactManagerConfig:
    guide_wire: GuideWireConfig | None = None
    markers: MarkerConfig | None = None
    stent: StentConfig | None = None
    external_files: list[ExternalArtifactConfig] = field(default_factory=list)


# ---- ArtifactManager ---------------------------------------------------------

class ArtifactManager:
    """Config-driven factory for generating artifacts from coronary labels.

    Handles artifact dependency order:
    1. Guide wire (requires coronary + cavity labels)
    2. Markers (requires guide wire centerline)
    3. Stent  (requires guide wire centerline + marker positions)
    4. External mesh files (loaded directly)
    """

    def __init__(self, config: ArtifactManagerConfig):
        self.config = config

    def generate_or_load(
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
        marker_centers: list[np.ndarray] | None = None
        
        if self.config.guide_wire is None:
            if self.config.markers is not None or self.config.stent is not None:
                raise ValueError(
                    "ArtifactManager: Cannot generate markers or stent without a guide wire. "
                    "Set guide_wire in ArtifactManagerConfig."
                )
            return artifacts  # empty list

        # 1. Guide wire
        gw_cfg = self.config.guide_wire
        gw_artifact, centerline_world = generate_guide_wire(
            coronary_label=coronary_label,
            cavity_label=cavity_label,
            affine=affine,
            wire_radius_mm=gw_cfg.radius_mm,
            mu=gw_cfg.mu,
        )
        artifacts.append(gw_artifact)
        
        if self.config.markers is None:
            if self.config.stent is not None:
                raise ValueError(
                    "ArtifactManager: Cannot generate stent without markers. "
                    "Set markers in ArtifactManagerConfig."
                )
            return artifacts  # only guide wire

        # 2. Markers (require guide wire centerline)
        mk_cfg = self.config.markers
        marker_artifacts = generate_markers(
            centerline_world=centerline_world,
            marker_distance_mm=mk_cfg.distance_mm,
            marker_radius_mm=mk_cfg.radius_mm,
            mu=mk_cfg.mu,
            margin_ratio=mk_cfg.margin_ratio,
        )
        artifacts.extend(marker_artifacts)
        marker_centers = [np.array(m.mesh.center) for m in marker_artifacts]
        
        if self.config.stent is None:
            return artifacts  # guide wire + markers

        # 3. Stent (requires guide wire centreline + marker positions)
        st_cfg = self.config.stent
        avg_cl_d = np.mean(np.linalg.norm(np.diff(centerline_world, axis=0), axis=1))
        idx_from_marker = int(st_cfg.margin_from_marker_mm / avg_cl_d)
        # Find nearest centreline indices to the two markers
        from scipy.spatial import KDTree
        tree = KDTree(centerline_world)
        _, idx0 = tree.query(marker_centers[0])   # distal marker
        _, idx1 = tree.query(marker_centers[1])   # proximal marker
        idx_proximal = int(min(idx0, idx1)) + idx_from_marker
        idx_distal = int(max(idx0, idx1)) - idx_from_marker
        if idx_distal - idx_proximal < 2:
            idx_distal = min(idx_proximal + 2, len(centerline_world) - 1)

        stent_artifact = generate_stent(
            centerline_world=centerline_world,
            idx_proximal=idx_proximal,
            idx_distal=idx_distal,
            stent_radius_mm=st_cfg.radius_mm,
            strut_radius_mm=st_cfg.strut_radius_mm,
            n_wires=st_cfg.n_wires,
            mu=st_cfg.mu,
        )
        artifacts.append(stent_artifact)

        # 4. External mesh files
        for ext_cfg in self.config.external_files:
            if not ext_cfg.path.exists():
                raise FileNotFoundError(f"External artifact mesh not found: {ext_cfg.path}")
            mesh = pv.read(ext_cfg.path)
            assert isinstance(mesh, pv.PolyData), f"External artifact mesh must be PolyData, got {type(mesh)}"
            name = ext_cfg.name if ext_cfg.name else ext_cfg.path.stem
            artifacts.append(Artifact(mesh=mesh, mu=ext_cfg.mu, name=name))

        return artifacts
