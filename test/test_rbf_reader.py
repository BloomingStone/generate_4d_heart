import numpy as np
import pyvista as pv
import pytest
import torch

from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.rotate_dsa.data_reader.rbf_reader import KDTreeRBF, invert_displacement_field
from generate_4d_heart.ssm import Landmark


def _build_landmark_for_rbf() -> Landmark:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mesh = pv.PolyData(points)

    deforms = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    return Landmark(mesh=mesh, deforms=deforms)


def test_invert_displacement_field_returns_zero_for_zero_field() -> None:
    u_forward = torch.zeros((1, 3, 8, 8, 8), dtype=torch.float32)

    inverse = invert_displacement_field(u_forward, n_iters=8)

    torch.testing.assert_close(inverse, torch.zeros_like(inverse), rtol=0, atol=1e-6)


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_invert_displacement_field_inverts_constant_translation(axis: int) -> None:
    u_forward = torch.zeros((1, 3, 12, 12, 12), dtype=torch.float32)
    u_forward[:, axis] = 1.0

    inverse = invert_displacement_field(u_forward, n_iters=16)

    expected = torch.zeros_like(u_forward)
    expected[:, axis] = -1.0
    torch.testing.assert_close(inverse, expected, rtol=0, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KDTreeRBF forward path uses CUDA stream.")
def test_kdtree_rbf_supports_phase_interpolation_and_cached_points() -> None:
    rbf = KDTreeRBF(
        _build_landmark_for_rbf(),
        sigma=1.0,
        chunk_size=1,
        radius_factor=10.0,
        k=2,
        n_ctrl_pts=2,
    ).to(torch.device("cuda:0"))

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    phase = CardiacPhase(0.25)

    direct_motion = rbf(phase, points)
    rbf.cache_points(points)
    cached_motion = rbf(phase)

    torch.cuda.synchronize()

    expected = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=direct_motion.device,
    )
    torch.testing.assert_close(direct_motion, cached_motion, rtol=0, atol=1e-3)
    torch.testing.assert_close(direct_motion, expected, rtol=0, atol=1e-3)


def test_kdtree_rbf_requires_cache_when_points_is_none() -> None:
    rbf = KDTreeRBF(_build_landmark_for_rbf(), k=1, n_ctrl_pts=2)

    with pytest.raises(AssertionError, match="cache_points"):
        rbf(CardiacPhase(0.1), None)


def test_kdtree_rbf_rejects_unsupported_points_type() -> None:
    rbf = KDTreeRBF(_build_landmark_for_rbf(), k=1, n_ctrl_pts=2)

    with pytest.raises(TypeError, match="Unsupported type"):
        rbf(CardiacPhase(0.1), points="bad points")  # type: ignore[arg-type]
