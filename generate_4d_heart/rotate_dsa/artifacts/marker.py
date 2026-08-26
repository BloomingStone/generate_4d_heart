from __future__ import annotations

import numpy as np
import pyvista as pv

from generate_4d_heart import MU_MARKER
from .artifact import Artifact


def generate_markers(
    centerline_world: np.ndarray,
    marker_distance_mm: float = 6.0,
    marker_radius_mm: float = 0.5,
    mu: float = MU_MARKER,
    start_pos_mm: float = 10.0,
) -> list[Artifact]:
    """Generate spherical marker artifacts along a guide wire centerline.

    The centerline is assumed to be ordered proximal -> distal (index 0 = near
    aorta).  Markers are placed starting from a random point in the middle
    portion and stepping backward (toward the proximal end) by a fixed number
    of index steps computed from ``marker_distance_mm`` and the median
    inter-point spacing.

    Parameters
    ----------
    centerline_world : np.ndarray
        (N, 3) centerline points in world coordinates, ordered proximal -> distal
        (from ``generate_guide_wire``).
    marker_distance_mm : float
        Desired physical distance between consecutive markers (mm).
    marker_radius_mm : float
        Radius of each marker sphere (mm).
    mu : float
        Linear attenuation coefficient (mm^-1).
    start_pos_mm : float | None
        Starting position of the first marker along the guide wire centerline.
        If None, the first marker will be placed at a random position along
        the centerline, with the second marker placed at distance_mm from the first.

    Returns
    -------
    list[Artifact]
        List of marker artifacts, each a sphere mesh.
    """
    n = len(centerline_world)
    if n < 2:
        raise ValueError("Centerline too short (< 2 points) to generate markers.")

    # 1. Average inter-point spacing (world coordinates = mm)
    diffs = np.diff(centerline_world, axis=0)
    dx = float(np.median(np.linalg.norm(diffs, axis=1))) if n > 1 else 1.0
    dp = max(1, int(round(marker_distance_mm / dx)))

    assert start_pos_mm >= 0, "start_pos_mm must be non-negative"
    diff_dist = np.linalg.norm(np.diff(centerline_world, axis=0), axis=1)
    cum_dist = np.concatenate(([0.0], np.cumsum(diff_dist)))
    idx1 = np.argmin(np.abs(cum_dist - start_pos_mm))
    point1 = centerline_world[idx1]
    idx2 = idx1 + dp
    if idx2 >= n:
        raise ValueError(f"start_pos_mm={start_pos_mm} is too close to the distal end of the centerline to place a second marker at distance {marker_distance_mm} mm.")
    point2 = centerline_world[idx2]

    artifacts: list[Artifact] = []
    for i, pos in enumerate([point1, point2]):
        sphere = pv.Sphere(
            radius=marker_radius_mm,
            center=pos.tolist(),
            theta_resolution=16,
            phi_resolution=16,
        )
        artifacts.append(Artifact(mesh=sphere, mu=mu, name=f"marker_{i}"))

    return artifacts
