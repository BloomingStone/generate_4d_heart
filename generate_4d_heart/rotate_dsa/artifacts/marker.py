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
    margin_ratio: float = 0.9,
    max_markers: int = 3,
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
    margin_ratio : float
        Fraction of the centerline length that the first marker may be sampled
        from.  ``margin_ratio = 0.9`` means the first marker is taken from the
        middle 90 % of the centerline, leaving 5 % excluded at each end.
    max_markers : int
        Maximum number of markers to generate.

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

    # 2. Margin: exclude `given_margin` points at each end, but at least dp*2
    given_margin = int((1.0 - margin_ratio) / 2.0 * n)
    margin = max(given_margin, dp * 2)
    if margin >= n // 2:            # fallback if margin is unreasonably large
        margin = max(1, n // 4)

    # 3. Point 1: random index in [margin, n - margin - 1]
    lo, hi = margin, n - margin - 1
    if lo > hi:
        lo, hi = 0, n - 1
    idx1 = int(np.random.randint(lo, hi + 1))
    point1 = centerline_world[idx1]

    # 4. Point 2: one step backward (proximal direction) from point 1
    idx2 = idx1 - dp
    if idx2 < margin:
        idx2 = margin
    point2 = centerline_world[idx2]

    # 5. Create sphere meshes
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
