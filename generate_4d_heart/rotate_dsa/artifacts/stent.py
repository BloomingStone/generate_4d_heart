from __future__ import annotations

import numpy as np
import pyvista as pv

from generate_4d_heart import MU_GUIDE_WIRE
from .artifact import Artifact


def generate_stent(
    centerline_world: np.ndarray,
    idx_proximal: int,
    idx_distal: int,
    stent_radius_mm: float = 1,
    strut_radius_mm: float = 0.15,
    n_wires: int = 6,
    mu: float = MU_GUIDE_WIRE,
) -> Artifact:
    """Generate a stent mesh between two points on a guide wire centerline.

    The stent is modelled as a tubular mesh of helical struts spanning the
    centerline segment from ``idx_proximal`` (near aorta) to ``idx_distal``
    (further from aorta).

    Parameters
    ----------
    centerline_world : np.ndarray
        (N, 3) centerline points in world coordinates, ordered proximal -> distal.
    idx_proximal : int
        Index of the proximal end of the stent (closer to aorta).
    idx_distal : int
        Index of the distal end of the stent (further from aorta).
    stent_radius_mm : float
        Radius of the stent tube (mm).
    strut_radius_mm : float
        Radius of each individual strut wire (mm).
    n_wires : int
        Number of helical wires (half go clockwise, half counter-clockwise).
    n_circumferential : int
        Number of points around each cross-section.
    mu : float
        Linear attenuation coefficient (mm^-1).

    Returns
    -------
    Artifact
        Stent artifact.
    """
    if idx_proximal >= idx_distal:
        raise ValueError(
            f"idx_proximal ({idx_proximal}) must be < idx_distal ({idx_distal})"
        )
    if len(centerline_world) <= idx_distal:
        raise ValueError(
            f"idx_distal ({idx_distal}) out of range (N={len(centerline_world)})"
        )

    # 1. Extract the centerline segment and resample uniformly
    seg_raw = np.asarray(
        centerline_world[idx_proximal : idx_distal + 1], dtype=np.float64
    )
    if len(seg_raw) < 2:
        raise ValueError("Stent segment too short.")
    # Resample to a fixed number of uniformly spaced points along the curve
    n_pts = max(len(seg_raw) * 2, 30)  # enough for smooth helix
    spline_curve = pv.Spline(seg_raw, n_points=n_pts)
    seg = np.array(spline_curve.points, dtype=np.float64)

    # 2. Compute local Frenet-like frames along the segment
    frames = _compute_frames(seg)
    pts = np.array([f[0] for f in frames])       # (n_pts, 3)
    normals = np.array([f[2] for f in frames])    # (n_pts, 3)
    binormals = np.array([f[3] for f in frames])  # (n_pts, 3)

    # 3. Generate helical strut paths directly
    strut_meshes: list[pv.PolyData] = []
    n_turns = 1.5
    # total angular progression along the full stent
    total_rad = n_turns * 2.0 * np.pi

    for w in range(n_wires):
        start_angle = 2.0 * np.pi * w / n_wires

        # Forward helix (clockwise)
        t = np.linspace(0.0, 1.0, n_pts)
        theta_fwd = start_angle + total_rad * t
        fwd = (
            pts
            + stent_radius_mm * (np.cos(theta_fwd)[:, None] * normals)
            + stent_radius_mm * (np.sin(theta_fwd)[:, None] * binormals)
        )
        strut_meshes.append(_path_to_tube(fwd, strut_radius_mm))

        # Backward helix (counter-clockwise)
        theta_bwd = start_angle - total_rad * t
        bwd = (
            pts
            + stent_radius_mm * (np.cos(theta_bwd)[:, None] * normals)
            + stent_radius_mm * (np.sin(theta_bwd)[:, None] * binormals)
        )
        strut_meshes.append(_path_to_tube(bwd, strut_radius_mm))

    # 5. Merge all strut tubes into a single mesh
    merged = strut_meshes[0].merge(strut_meshes[1:])
    # Clean up duplicate points introduced by merging
    merged = merged.clean()
    merged = merged.triangulate()

    return Artifact(mesh=merged, mu=mu, name="stent")


# ── helpers ─────────────────────────────────────────────────────────

def _compute_frames(
    seg: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Compute (point, tangent, normal, binormal) frames along a 3-D curve."""
    n = len(seg)
    tangents = np.zeros_like(seg)
    tangents[0] = seg[1] - seg[0]
    tangents[-1] = seg[-1] - seg[-2]
    for i in range(1, n - 1):
        tangents[i] = seg[i + 1] - seg[i - 1]

    # Normalize tangents
    tnorms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tnorms[tnorms == 0] = 1.0
    tangents /= tnorms

    # Parallel transport normals to avoid twist
    normals = np.zeros_like(seg)
    # Initial normal: any vector perpendicular to first tangent
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(tangents[0], ref)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normals[0] = np.cross(tangents[0], ref)
    nnorm = np.linalg.norm(normals[0])
    if nnorm > 0:
        normals[0] /= nnorm
    else:
        normals[0] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Parallel transport for subsequent normals
    for i in range(1, n):
        t_prev = tangents[i - 1]
        t_cur = tangents[i]
        n_prev = normals[i - 1]
        # Rotation that aligns t_prev to t_cur
        axis = np.cross(t_prev, t_cur)
        anorm = np.linalg.norm(axis)
        if anorm < 1e-12:
            normals[i] = n_prev
        else:
            axis /= anorm
            cos_a = np.dot(t_prev, t_cur)
            # Rodrigues' rotation formula
            n_rot = (
                n_prev * cos_a
                + np.cross(axis, n_prev) * np.sqrt(max(0.0, 1.0 - cos_a * cos_a))
                + axis * np.dot(axis, n_prev) * (1.0 - cos_a)
            )
            normals[i] = n_rot
        nnorm = np.linalg.norm(normals[i])
        if nnorm > 0:
            normals[i] /= nnorm
        else:
            normals[i] = n_prev.copy()

    binormals = np.cross(tangents, normals)
    binorms = np.linalg.norm(binormals, axis=1, keepdims=True)
    binorms[binorms == 0] = 1.0
    binormals /= binorms

    return [
        (seg[i], tangents[i], normals[i], binormals[i]) for i in range(n)
    ]


def _path_to_tube(points: np.ndarray, radius: float) -> pv.PolyData:
    """Create a thin tube mesh along a 3-D path."""
    if len(points) < 2:
        # Single point → degenerate; return a tiny sphere to avoid breakage
        return pv.Sphere(radius=radius, center=points[0].tolist())
    spline = pv.Spline(points, n_points=len(points) * 2)
    return spline.tube(radius=radius)
