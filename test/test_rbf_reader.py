from pathlib import Path
import shutil

import numpy as np
import nibabel as nib
import pyvista as pv
import pytest
import torch

from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.rotate_dsa.data_reader.rbf_reader import KDTreeRBF, invert_displacement_field
from generate_4d_heart.ssm import Landmark
from generate_4d_heart.rotate_dsa.artifacts import ArtifactManager
from generate_4d_heart.rotate_dsa.data_reader import RBFReader
from generate_4d_heart.rotate_dsa.data_reader.data_reader import cleanup_gpu_memory


from utils import SimulatorName, TEST_DATA_ROOT_DIR, simulator_factory




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
    phase = CardiacPhase.from_time(0.25, cardiac_cycle_time=1.0)

    direct_motion = rbf(phase, points)
    rbf.cache_points(points)
    cached_motion = rbf(phase)

    torch.cuda.synchronize()

    expected = torch.tensor(
        [
            [0.25, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=direct_motion.device,
    )
    torch.testing.assert_close(direct_motion, cached_motion, rtol=0, atol=1e-3)
    torch.testing.assert_close(direct_motion, expected, rtol=0, atol=1e-3)


def test_kdtree_rbf_requires_cache_when_points_is_none() -> None:
    rbf = KDTreeRBF(_build_landmark_for_rbf(), k=1, n_ctrl_pts=2)

    with pytest.raises(AssertionError, match="cache_points"):
        rbf(CardiacPhase.from_time(0.1, cardiac_cycle_time=1.0), None)


def test_kdtree_rbf_rejects_unsupported_points_type() -> None:
    rbf = KDTreeRBF(_build_landmark_for_rbf(), k=1, n_ctrl_pts=2)

    with pytest.raises(TypeError, match="Unsupported type"):
        rbf(CardiacPhase.from_time(10, cardiac_cycle_time=1.0), points="bad points")  # type: ignore[arg-type]



# TODO FIX: coronary 输出为 空
@pytest.mark.parametrize("simulator_name", (
    SimulatorName.HU2MU_CONTRAST, SimulatorName.STATIC_IODINE_CONTRAST, SimulatorName.CTA_RESET_CONTRAST, SimulatorName.FLOW_CONTRAST
))
def test_reader_and_save_artifacts_pig_drr_chest_background(
    output_root_dir: Path,
    simulator_name: SimulatorName,
    artifact_manager: ArtifactManager,
):
    """Integration test with chest CTA data + background CT, using pig_data_drr."""
    # Clean up GPU memory from any previous test failure that may have leaked.
    # This prevents CuPy OOM when this test allocates large background volumes.
    cleanup_gpu_memory()

    chest_dir = TEST_DATA_ROOT_DIR / "chest_data"
    assert chest_dir.exists(), f"chest_data directory not found: {chest_dir}"

    reader = RBFReader(
        volume_nii=chest_dir / "image.nii.gz",
        cavity_nii=chest_dir / "cavity.nii.gz",
        coronary_nii=chest_dir / "coronary.nii.gz",
        contrast_simulator=simulator_factory(simulator_name),
        artifact_manager=artifact_manager,
        background_nii=chest_dir / "background.nii.gz",
        centering_mode = "image_center"
    )

    output_dir = output_root_dir / "readers_output" / "rbf_reader_full_save" / f"pig_drr_chest_background_{simulator_name}_LCA"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = reader.get_phase_0_data("LCA")
    result.save(
        output_dir,
        save_nii_type=("cavity", "coronary", "volume"),
        save_artifacts=True,
        save_central_line=True,
        save_mesh=True,
        save_meta = True,
        coordinate_system="world",
    )

    # ── Regression guard: exported NIfTI files must contain non-empty values ──
    # The combined volume (static layers in Centering space) must align with the
    # resampled dynamic cavity/coronary labels; otherwise labels come out empty.
    from nibabel.nifti1 import Nifti1Image

    def _load_arr(path: Path) -> np.ndarray:
        img = nib.load(path)
        assert isinstance(img, Nifti1Image)
        return np.asanyarray(img.dataobj)

    for nii_name in ("cavity_label.nii.gz", "coronary_label.nii.gz", "volume_with_contrast.nii.gz"):
        nii_path = output_dir / nii_name
        assert nii_path.exists(), f"Expected {nii_name} to be saved"
        arr = _load_arr(nii_path)
        assert arr.size > 0
        assert float(arr.max()) > 0, f"{nii_name} is empty (max value = 0)"

    cav = _load_arr(output_dir / "cavity_label.nii.gz")
    cor = _load_arr(output_dir / "coronary_label.nii.gz")
    vol = _load_arr(output_dir / "volume_with_contrast.nii.gz")
    assert (cav > 0).sum() > 0, "cavity_label.nii.gz has no foreground voxels"
    assert (cor > 0).sum() > 0, "coronary_label.nii.gz has no foreground voxels"
    # Coronary label must coincide with volume foreground (labels & volume aligned)
    assert (vol[cor > 0] > 0).any(), "coronary label does not overlap any volume foreground voxel"
    
    
