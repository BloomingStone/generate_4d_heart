from __future__ import annotations

from collections import deque
from itertools import product

import numpy as np
import pyvista as pv
from skimage.morphology import skeletonize
from scipy.spatial import KDTree

from generate_4d_heart import MU_GUIDE_WIRE, VesselLabel
from .artifact import Artifact


def generate_guide_wire(
    coronary_label: np.ndarray,
    cavity_label: np.ndarray,
    affine: np.ndarray,
    wire_radius_mm: float = 0.2,
    mu: float = MU_GUIDE_WIRE,
) -> tuple[Artifact, np.ndarray]:
    """Generate a guide wire artifact from a single coronary label.

    The wire follows the longest path along the coronary skeleton,
    starting near the aorta.

    Parameters
    ----------
    coronary_label : np.ndarray
        3D binary/uint8 mask of a single coronary artery (LCA or RCA).
    cavity_label : np.ndarray
        3D uint8 cavity label map (used to locate the aorta).
    affine : np.ndarray
        4x4 affine matrix mapping voxel indices -> world coordinates.
    wire_radius_mm : float
        Radius of the guide wire tube in mm.
    mu : float
        Linear attenuation coefficient in mm^-1.

    Returns
    -------
    tuple[Artifact, np.ndarray]
        (guide_wire_artifact, centerline_world)
        - centerline_world: (N, 3) array of centerline points in world coordinates.
    """
    coronary_bool = coronary_label.squeeze().astype(bool)
    cavity_np = cavity_label.squeeze()

    # 1. Skeletonize
    skeleton = skeletonize(coronary_bool, method="lee").astype(bool)

    # 2. Find start voxel (nearest skeleton voxel to aorta)
    start_point = _resolve_start_voxel(cavity_np, skeleton)

    # 3. BFS to build tree and find deepest leaf
    coords = np.array(np.where(skeleton)).T  # (M, 3)
    coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
    n = len(coords)

    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        ci = coords[i]
        for di, dj, dk in product((-1, 0, 1), repeat=3):
            if di == 0 and dj == 0 and dk == 0:
                continue
            nb_key = (int(ci[0] + di), int(ci[1] + dj), int(ci[2] + dk))
            j = coord_to_idx.get(nb_key)
            if j is not None and j > i:
                adj[i].append(j)
                adj[j].append(i)

    start_idx = coord_to_idx[tuple(start_point)]
    parent = np.full(n, -1, dtype=int)
    depth_arr = np.full(n, -1, dtype=int)
    parent[start_idx] = start_idx
    depth_arr[start_idx] = 0
    q: deque[int] = deque([start_idx])
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if parent[nb] == -1:
                parent[nb] = cur
                depth_arr[nb] = depth_arr[cur] + 1
                q.append(nb)

    # Find leaf nodes
    has_child = np.zeros(n, dtype=bool)
    for i in range(n):
        p = parent[i]
        if p != -1 and p != i:
            has_child[p] = True
    leaves = [i for i in range(n) if parent[i] != -1 and not has_child[i]]

    if not leaves:
        raise ValueError("No leaf nodes found in coronary skeleton -- cannot generate guide wire.")

    # Deepest leaf -> longest path
    chosen_leaf = int(leaves[np.argmax([depth_arr[l] for l in leaves])])
    path_indices: list[int] = []
    cur = chosen_leaf
    while cur != start_idx:
        path_indices.append(cur)
        cur = int(parent[cur])
    path_indices.append(start_idx)

    # 4. Centerline in voxel coords (ordered proximal -> distal: index 0 = near aorta)
    centerline_voxel = np.array([coords[idx] for idx in reversed(path_indices)], dtype=np.float64)

    # 5. Convert to world coordinates
    centerline_world = _apply_affine_np(centerline_voxel, affine)

    # 6. Smooth centerline to remove jagged skeleton artifacts
    centerline_world = _smooth_centerline(centerline_world, window=7)

    # 7. Subsample if too dense (avoid huge mesh)
    step = max(1, len(centerline_world) // 500)
    centerline_sub = centerline_world[::step]
    if len(centerline_sub) < 2:
        centerline_sub = centerline_world

    # 8. Create tube mesh (centerline_sub is already ordered proximal -> distal)
    spline = pv.Spline(centerline_sub, n_points=len(centerline_sub) * 2)
    tube_mesh = spline.tube(radius=wire_radius_mm)

    artifact = Artifact(mesh=tube_mesh, mu=mu, name="guide_wire")
    return artifact, centerline_world


def _resolve_start_voxel(cavity: np.ndarray, skeleton_mask: np.ndarray) -> tuple[int, int, int]:
    """Find the skeleton voxel nearest to the aorta region."""
    if not skeleton_mask.any():
        raise ValueError("Empty skeleton -- cannot resolve start voxel.")

    aorta_mask = (cavity == int(VesselLabel.AORTA))
    if not aorta_mask.any():
        raise ValueError("Empty aorta region -- cannot resolve start voxel.")

    return _nearest_skeleton_voxel_to_region(aorta_mask, skeleton_mask)


def _nearest_skeleton_voxel_to_region(
    region_mask: np.ndarray,
    skeleton_mask: np.ndarray,
) -> tuple[int, int, int]:
    skel_coords = np.argwhere(skeleton_mask)
    if len(skel_coords) == 0:
        raise ValueError("No skeleton voxels found.")

    region_coords = np.argwhere(region_mask)
    if len(region_coords) == 0:
        return (int(skel_coords[0][0]), int(skel_coords[0][1]), int(skel_coords[0][2]))

    tree = KDTree(skel_coords.astype(np.float32))
    dists, nearest_idxs = tree.query(region_coords.astype(np.float32), k=1)
    best = int(np.argmin(dists))
    nearest = skel_coords[nearest_idxs[best]]
    return (int(nearest[0]), int(nearest[1]), int(nearest[2]))


def _smooth_centerline(pts: np.ndarray, window: int = 7) -> np.ndarray:
    """Smooth a centerline by 1D uniform filter per axis."""
    if len(pts) < window or window < 3:
        return pts
    from scipy.ndimage import uniform_filter1d
    return np.stack([
        uniform_filter1d(pts[:, i], size=window, mode='nearest')
        for i in range(3)
    ], axis=1)


def _apply_affine_np(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine to (N, 3) points in voxel -> world conversion."""
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.concatenate([points, ones], axis=1)  # (N, 4)
    world = (affine @ homogeneous.T).T  # (N, 4)
    return world[:, :3]
