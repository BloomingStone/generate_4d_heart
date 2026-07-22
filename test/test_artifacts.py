from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest
import torch
import matplotlib.pyplot as plt

from generate_4d_heart import MU_GUIDE_WIRE, MU_MARKER
from generate_4d_heart.rotate_dsa.data_reader.data_reader import (
    load_nifti,
    separate_coronary,
)
from generate_4d_heart.rotate_dsa.rotate_drr import CArmGeometry, TorchDRR, RotatedParameters
from generate_4d_heart.rotate_dsa.artifacts import (
    Artifact,
    generate_guide_wire,
    generate_markers,
    generate_stent,
    ArtifactManager,
    ArtifactManagerConfig,
    GuideWireConfig,
    MarkerConfig,
    StentConfig,
)

from generate_4d_heart.rotate_dsa.torch3d_label_renderer import Torch3DLabelRenderer


# ---- Fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def lca_data(volume_dvf_dir: Path) -> dict:
    """Load coronary + cavity NIfTI and separate LCA.

    Returns a dict with keys: lca_label, cavity_label, affine.
    All arrays are 3D numpy (squeezed).
    """
    coronary_path = volume_dvf_dir / "coronary.nii.gz"
    cavity_path = volume_dvf_dir / "cavity.nii.gz"

    coronary, affine = load_nifti(coronary_path, is_label=True)  # (1,1,D,H,W)
    cavity, _ = load_nifti(cavity_path, is_label=True)

    assert affine is not None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lca, _ = separate_coronary(coronary, device=device)

    return {
        "lca_label": lca.squeeze().cpu().numpy().astype(bool),
        "cavity_label": cavity.squeeze().cpu().numpy(),
        "affine": affine,
    }


# ---- Test guide wire generation ---------------------------------------------

class TestGuideWireGeneration:
    def test_generate_returns_artifact_and_centerline(self, lca_data):
        art, cl = generate_guide_wire(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        assert isinstance(art, Artifact)
        assert art.name == "guide_wire"
        assert art.mu == MU_GUIDE_WIRE
        assert isinstance(cl, np.ndarray)
        assert cl.ndim == 2 and cl.shape[1] == 3
        assert cl.shape[0] >= 2, "Centerline should have at least 2 points"

    def test_guide_wire_mesh_is_valid(self, lca_data):
        art, _ = generate_guide_wire(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        mesh = art.mesh
        assert mesh.n_points > 0, "Mesh should have points"
        assert mesh.n_cells > 0, "Mesh should have cells"
        # Tube mesh uses triangle strips; each centerline segment -> 1 strip
        assert mesh.n_cells >= 10, (
            f"Guide wire mesh seems too small ({mesh.n_cells} strips)"
        )
        assert mesh.n_points > 1000, (
            f"Guide wire mesh too few points ({mesh.n_points})"
        )

    def test_custom_wire_radius(self, lca_data):
        art, _ = generate_guide_wire(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
            wire_radius_mm=0.5,
        )
        # Sanity: bounding box should reflect the larger radius
        bounds = art.mesh.bounds
        extents = np.array([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ])
        assert np.any(extents > 0.01), "Mesh bounding box is degenerate"

    def test_centerline_in_world_coordinates(self, lca_data):
        """Centerline should be far from origin (since affine shifts voxel->world)."""
        _, cl = generate_guide_wire(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        # World coordinates of a CT coronary are typically > 100 mm from origin
        mean_pos = np.mean(np.abs(cl))
        assert mean_pos > 10.0, (
            f"Centerline seems to be in voxel space (mean abs pos = {mean_pos:.1f})"
        )

    def test_empty_coronary_raises(self, lca_data):
        empty_label = np.zeros_like(lca_data["lca_label"])
        with pytest.raises((ValueError, IndexError)):
            generate_guide_wire(
                coronary_label=empty_label,
                cavity_label=lca_data["cavity_label"],
                affine=lca_data["affine"],
            )


# ---- Test marker generation -------------------------------------------------

class TestMarkerGeneration:
    def test_generate_markers_from_real_centerline(self, lca_data):
        _, cl = generate_guide_wire(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        markers = generate_markers(
            centerline_world=cl,
            marker_distance_mm=6.0,
            marker_radius_mm=0.5,
        )
        assert len(markers) >= 1, "Should generate at least 1 marker"
        for m in markers:
            assert isinstance(m, Artifact)
            assert m.mu == MU_MARKER
            assert m.name.startswith("marker_")
            assert m.mesh.n_points > 0
            assert m.mesh.n_cells > 0

    def test_markers_near_centerline(self, lca_data):
        """Each marker center should be very close to the centerline path."""
        _, cl = generate_guide_wire(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        markers = generate_markers(
            centerline_world=cl,
            marker_distance_mm=10.0,
            marker_radius_mm=0.5,
        )
        for m in markers:
            center = np.array(m.mesh.center)
            dists = np.linalg.norm(cl - center, axis=1)
            min_dist = dists.min()
            assert min_dist < 1.0, (
                f"Marker center {center} is {min_dist:.2f}mm from centerline"
            )

    def test_custom_mu(self, lca_data):
        _, cl = generate_guide_wire(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        custom_mu = 0.75
        markers = generate_markers(
            centerline_world=cl,
            marker_distance_mm=8.0,
            marker_radius_mm=0.3,
            mu=custom_mu,
        )
        assert len(markers) >= 1
        for m in markers:
            assert m.mu == custom_mu

    def test_always_two_markers(self, lca_data):
        _, cl = generate_guide_wire(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        for _ in range(5):  # run multiple times to verify
            markers = generate_markers(
                centerline_world=cl,
                marker_distance_mm=2.0,
                marker_radius_mm=0.3,
            )
            assert len(markers) == 2, (
                f"Expected 2 markers, got {len(markers)}"
            )

    def test_short_centerline_raises(self):
        with pytest.raises(ValueError, match="Centerline too short"):
            generate_markers(
                centerline_world=np.ones((1, 3)),
                marker_distance_mm=6.0,
            )



# ---- Test Stent generation ---------------------------------------------------

class TestStentGeneration:
    def test_generate_stent_synthetic(self):
        """Stent on a straight centerline segment."""
        cl = np.zeros((30, 3), dtype=np.float64)
        cl[:, 2] = np.linspace(0, 30, 30)
        stent = generate_stent(cl, 5, 25, stent_radius_mm=2.0, strut_radius_mm=0.15)
        assert stent.name == "stent"
        assert stent.mesh.n_points > 0
        assert stent.mesh.n_cells > 0

    def test_stent_uses_large_portion(self, lca_data):
        """Stent on a real coronary centerline."""
        _, cl = generate_guide_wire(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        idx_p, idx_d = len(cl) // 4, len(cl) * 3 // 4
        stent = generate_stent(cl, idx_p, idx_d, stent_radius_mm=1.2, strut_radius_mm=0.1)
        assert stent.mesh.n_points > 1000

    def test_stent_proximal_before_distal_raises(self):
        cl = np.zeros((10, 3))
        with pytest.raises(ValueError, match="idx_proximal"):
            generate_stent(cl, 7, 3)

    def test_stent_idx_out_of_range_raises(self):
        cl = np.zeros((10, 3))
        with pytest.raises(ValueError, match="idx_distal"):
            generate_stent(cl, 0, 99)

    def test_stent_too_short_segment(self, lca_data):
        """Stent with identical markers should raise or produce degenerate mesh."""
        _, cl = generate_guide_wire(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        stent = generate_stent(cl, 5, 6, stent_radius_mm=1.0, strut_radius_mm=0.05)
        assert stent.mesh.n_points > 0


# ---- Test ArtifactManager with stent -----------------------------------------

class TestArtifactManagerStent:
    def test_manager_guide_wire_markers_stent(self, lca_data):
        """Full pipeline: guide wire → markers → stent between markers."""
        config = ArtifactManagerConfig(
            guide_wire=GuideWireConfig(radius_mm=0.3),
            markers=MarkerConfig(distance_mm=10.0, radius_mm=0.4),
            stent=StentConfig(radius_mm=1.5, strut_radius_mm=0.1, n_wires=8),
        )
        manager = ArtifactManager(config)
        artifacts = manager.generate_or_load(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        names = [a.name for a in artifacts]
        assert "guide_wire" in names
        assert any(n.startswith("marker_") for n in names)
        assert "stent" in names
        stent = [a for a in artifacts if a.name == "stent"][0]
        assert stent.mesh.n_points > 1000

    def test_manager_stent_without_markers_raises(self, lca_data):
        config = ArtifactManagerConfig(
            guide_wire=GuideWireConfig(),
            markers=None,
            stent=StentConfig(),
        )
        manager = ArtifactManager(config)
        with pytest.raises(ValueError, match="ArtifactManager: Cannot generate stent without markers. Set markers in ArtifactManagerConfig."):
            manager.generate_or_load(
                coronary_label=lca_data["lca_label"],
                cavity_label=lca_data["cavity_label"],
                affine=lca_data["affine"],
            )

    def test_manager_stent_without_guide_wire_raises(self, lca_data):
        config = ArtifactManagerConfig(
            guide_wire=None,
            markers=MarkerConfig(),
            stent=StentConfig(),
        )
        manager = ArtifactManager(config)
        with pytest.raises(ValueError, match="ArtifactManager: Cannot generate markers or stent without a guide wire. Set guide_wire in ArtifactManagerConfig."):
            manager.generate_or_load(
                coronary_label=lca_data["lca_label"],
                cavity_label=lca_data["cavity_label"],
                affine=lca_data["affine"],
            )

# ---- Test ArtifactManager ---------------------------------------------------

class TestArtifactManager:
    def test_manager_guide_wire_only(self, lca_data):
        config = ArtifactManagerConfig(
            guide_wire=GuideWireConfig(radius_mm=0.3),
            markers=None,
        )
        manager = ArtifactManager(config)
        artifacts = manager.generate_or_load(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        assert len(artifacts) == 1
        assert artifacts[0].name == "guide_wire"

    def test_manager_guide_wire_with_markers(self, lca_data):
        config = ArtifactManagerConfig(
            guide_wire=GuideWireConfig(),
            markers=MarkerConfig(distance_mm=8.0, radius_mm=0.4),
        )
        manager = ArtifactManager(config)
        artifacts = manager.generate_or_load(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        names = [a.name for a in artifacts]
        assert "guide_wire" in names
        marker_names = [n for n in names if n.startswith("marker_")]
        assert len(marker_names) >= 1

    def test_manager_markers_without_guide_wire_raises(self, lca_data):
        config = ArtifactManagerConfig(
            guide_wire=None,
            markers=MarkerConfig(),
        )
        manager = ArtifactManager(config)
        with pytest.raises(ValueError, match="ArtifactManager: Cannot generate markers or stent without a guide wire. Set guide_wire in ArtifactManagerConfig."):
            manager.generate_or_load(
                coronary_label=lca_data["lca_label"],
                cavity_label=lca_data["cavity_label"],
                affine=lca_data["affine"],
            )

    def test_manager_no_config(self, lca_data):
        config = ArtifactManagerConfig()
        manager = ArtifactManager(config)
        artifacts = manager.generate_or_load(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        assert len(artifacts) == 0


# ---- Test Torch3DLabelRenderer --------------------------------------------------

@pytest.fixture(scope="module")
def renderer() -> Torch3DLabelRenderer:
    c_arm = CArmGeometry()
    return Torch3DLabelRenderer(c_arm)


@pytest.fixture(scope="module")
def drr_pose() -> tuple[torch.Tensor, torch.Tensor]:
    """Get R, T from a TorchDRR at frame 0 (proven working camera setup)."""
    drr = TorchDRR(
        c_arm_cfg=CArmGeometry(),
        rotate_cfg=RotatedParameters(total_frame=2, fps=1, angular_velocity=0, alpha_start=30),
    )
    return drr.get_R_T_at_frame(0)

@pytest.fixture(scope="module")
def artifact_output_dir(output_root_dir: Path) -> Path:
    out = output_root_dir / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    return out

@staticmethod
def _save_render_png(path: Path, arr: np.ndarray, title: str = ""):
    """Save a 2D line-integral array as a grayscale PNG visualization."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap="hot", aspect="auto")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label="line integral (mm⁻¹·mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Torch3DLabelRenderer")
class TestArtifactRendering:
    def test_render_empty_artifacts(self, renderer, drr_pose, artifact_output_dir):
        R, T = drr_pose
        result = renderer.render_line_integral([], R, T)
        assert result.shape == (renderer.height, renderer.width)
        assert result.sum() == 0.0
        _save_render_png(
            artifact_output_dir / "render_empty.png", result, "Empty artifacts"
        )

    def test_render_sphere_at_origin(
        self, renderer, drr_pose, artifact_output_dir
    ):
        """Place a sphere at the origin and verify it's visible."""
        sphere_radius_mm = 5.0
        sphere = pv.Sphere(
            radius=sphere_radius_mm,
            center=(0.0, 0.0, 0.0),
            theta_resolution=32,
            phi_resolution=32,
        )
        artifact = Artifact(mesh=sphere, mu=1.0, name="test_sphere")

        R, T = drr_pose
        result = renderer.render_line_integral([artifact], R, T)

        # Sphere should project somewhere in the image
        assert result.max() > 0.0, "Sphere should be visible in the render"

        # Corner pixel should be background
        assert result[0, 0] == 0.0, "Corner pixel should be background"

        # Maximum path length bounded by sphere diameter
        expected_max = 2.0 * sphere_radius_mm
        assert result.max() <= expected_max * 1.5, (
            f"Max line integral {result.max():.2f} too high"
        )

        # Save outputs
        sphere.save(artifact_output_dir / "sphere_mesh.vtp", binary=True)
        _save_render_png(
            artifact_output_dir / "render_sphere.png",
            result,
            f"Sphere (r={sphere_radius_mm}mm, mu=1.0)",
        )

    def test_render_two_spheres_sum_equals_individual(
        self, renderer, drr_pose, artifact_output_dir
    ):
        """
        Combined render of two spheres should equal sum of individual renders.
        """
        
        sphere1 = pv.Sphere(radius=3.0, center=(0.0, 0.0, 0.0),
                            theta_resolution=16, phi_resolution=16)
        sphere2 = pv.Sphere(radius=2.0, center=(10.0, 10.0, 10.0),
                            theta_resolution=16, phi_resolution=16)

        art1 = Artifact(mesh=sphere1, mu=0.5, name="s1")
        art2 = Artifact(mesh=sphere2, mu=1.0, name="s2")

        R, T = drr_pose

        combined = renderer.render_line_integral([art1, art2], R, T)
        single1 = renderer.render_line_integral([art1], R, T)
        single2 = renderer.render_line_integral([art2], R, T)

        assert np.allclose(combined, single1 + single2, atol=1e-6), (
            "Combined render should equal sum of individual renders"
        )

        # Save outputs
        sphere1.save(artifact_output_dir / "sphere1_mesh.vtp", binary=True)
        sphere2.save(artifact_output_dir / "sphere2_mesh.vtp", binary=True)
        _save_render_png(
            artifact_output_dir / "render_two_spheres_combined.png",
            combined,
            "Two spheres (combined)",
        )
        _save_render_png(
            artifact_output_dir / "render_two_spheres_single1.png",
            single1,
            "Sphere 1 (r=3mm, mu=0.5)",
        )
        _save_render_png(
            artifact_output_dir / "render_two_spheres_single2.png",
            single2,
            "Sphere 2 (r=2mm, mu=1.0)",
        )

    def test_line_integral_linear_in_mu(
        self, renderer, drr_pose, artifact_output_dir
    ):
        """Line integral should scale linearly with mu."""
        sphere = pv.Sphere(radius=3.0, center=(0.0, 0.0, 0.0),
                           theta_resolution=16, phi_resolution=16)

        art_a = Artifact(mesh=sphere.copy(), mu=1.0, name="a")
        art_b = Artifact(mesh=sphere.copy(), mu=2.0, name="b")

        R, T = drr_pose

        res_a = renderer.render_line_integral([art_a], R, T)
        res_b = renderer.render_line_integral([art_b], R, T)

        assert np.allclose(res_b, 2.0 * res_a, atol=1e-6), (
            "Line integral should scale linearly with mu"
        )

        # Save outputs
        _save_render_png(
            artifact_output_dir / "render_mu1.png",
            res_a,
            "Sphere (mu=1.0)",
        )
        _save_render_png(
            artifact_output_dir / "render_mu2.png",
            res_b,
            "Sphere (mu=2.0)",
        )


# ---- Test Artifact dataclass (sanity) ---------------------------------------

class TestArtifactDataclass:
    def test_create_artifact(self):
        sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
        art = Artifact(mesh=sphere, mu=0.5, name="test")
        assert art.mu == 0.5
        assert art.name == "test"
        assert art.mesh.n_points > 0

    def test_default_name(self):
        sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
        art = Artifact(mesh=sphere, mu=1.0)
        assert art.name == ""


class TestArtifactIntergration:
    def _compute_centering_offset(self, lca_data: dict) -> np.ndarray:
        """Compute translation to move world-coord artifacts to centering coords.

        The centering coordinate system places the coronary AABB center
        (Z at volume centre) at the origin — the same system used by
        ``Coronary.mesh_centering`` and the DRR camera.
        """
        label = lca_data["lca_label"]
        affine = lca_data["affine"]
        coords = np.argwhere(label)
        aabb_min = coords.min(axis=0)
        aabb_max = coords.max(axis=0)
        image_center_z = np.array(label.shape)[2] // 2
        center_voxel = np.array([
            (aabb_min[0] + aabb_max[0]) / 2.0,
            (aabb_min[1] + aabb_max[1]) / 2.0,
            image_center_z,
        ])
        B = affine[:3, :3]
        centering_T = -(B @ center_voxel)
        original_T = affine[:3, 3]
        return centering_T - original_T

    def test_generate_and_render_artifacts(
        self, lca_data, renderer, drr_pose, artifact_output_dir, artifact_manager
    ):
        """Generate guide wire + markers + stent, then render them (centered)."""
        artifacts = artifact_manager.generate_or_load(
            coronary_label=lca_data["lca_label"],
            cavity_label=lca_data["cavity_label"],
            affine=lca_data["affine"],
        )
        assert len(artifacts) >= 1
        names = [a.name for a in artifacts]
        assert "guide_wire" in names
        assert any(n.startswith("marker_") for n in names)
        assert "stent" in names

        # Translate artifacts from world → centering coordinates
        offset = self._compute_centering_offset(lca_data)
        for art in artifacts:
            art.mesh.points = art.mesh.points + offset

        R, T = drr_pose
        result = renderer.render_line_integral(artifacts, R, T)

        assert result.max() > 0.0, (
            "Rendered artifacts should be visible after centering"
        )

        # Save outputs
        for art in artifacts:
            art.mesh.save(artifact_output_dir / f"{art.name}.vtp", binary=True)
        _save_render_png(
            artifact_output_dir / "render_all.png",
            result,
            "Guide wire + markers + stent (centered)",
        )

        for art in artifacts:
            rendered = renderer.render_line_integral([art], R, T)
            _save_render_png(
                artifact_output_dir / f"render_all_{art.name}.png",
                rendered,
                f"Artifact: {art.name} (mu={art.mu})",
            )

