from typing import Literal, cast
from pathlib import Path

import pyvista as pv
import pytest
import torch
import numpy as np
from tqdm import tqdm

from generate_4d_heart.rotate_dsa.data_reader.data_reader import DataReader, get_mesh_in_world, CoronaryType, cleanup_gpu_memory
from generate_4d_heart.rotate_dsa.artifacts import (
    ArtifactManager, ArtifactManagerConfig, GuideWireConfig, MarkerConfig, StentConfig
)
from generate_4d_heart.rotate_dsa.data_reader import RBFReader
from generate_4d_heart.rotate_dsa.contrast_simulator import IdentityContrast, CtaResetContrast
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.saver import save_gif, save_nii, save_png

from utils import reader_factory, ReaderName, SimulatorName, OUTPUT_ROOT_DIR, TEST_DATA_ROOT_DIR, simulator_factory


def _read_and_save(reader: DataReader, output_dir: Path, coronary_type: Literal["LCA", "RCA"] = "LCA") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # delete result in output_dir if exists
    for p in output_dir.iterdir():
        if p.is_dir():
            for f in p.iterdir():
                f.unlink()
            p.rmdir()
        else:
            p.unlink()
    
    # for _ in tqdm(range(3), desc="Generating 3 random frames"):
    #     data = reader.get_data(CardiacPhase(random()), coronary_type)
    #     data.save(output_dir)
    
    F = 15
    W, H, D = reader.volume_size
    frames_w = []
    frames_h = []
    frames_d = []

    volume = None
    
    plotter_original = pv.Plotter(off_screen=True)
    plotter_original.open_gif(output_dir / "animation_original.gif")
    
    plotter_centering = pv.Plotter(off_screen=True)
    plotter_centering.open_gif(output_dir / "animation_centering.gif")
    
    test_res_original = []
    test_res_centering = []
    for phase_idx in tqdm(range(F), desc=f"Generating {F} continous frames in one cardiac cycle"):
        phase = CardiacPhase.from_index(phase_idx, F)

        data = reader.get_dynamic_data(phase, coronary_type)
        volume = data.coronary.volume.data[0, 0]

        frames_w.append(volume[W//2])
        frames_h.append(volume[:, H//2])
        frames_d.append(volume[:, :, D//2])

        mesh_original_from_label = get_mesh_in_world(data.coronary.label.data, data.coronary.original_affine, max_points=None)
        test_res_original.append(np.allclose(mesh_original_from_label.bounds, data.coronary.mesh_original.bounds, atol=1, rtol=0.1))

        mesh_centering_from_label = get_mesh_in_world(data.coronary.label.data, data.coronary.centering_affine, max_points=None)
        test_res_centering.append(np.allclose(mesh_centering_from_label.bounds, data.coronary.mesh_centering.bounds, atol=1, rtol=0.1))
        
        plotter_original.clear()
        plotter_original.add_mesh(data.coronary.mesh_original, color="lightgray", opacity=0.5)
        plotter_original.add_mesh(data.coronary.get_coronary_central_line("world"), color="red", line_width=5)
        plotter_original.add_mesh(mesh_original_from_label, opacity=0.5)
        plotter_original.show_bounds()  #type: ignore
        plotter_original.write_frame()
        
        plotter_centering.clear()
        plotter_centering.add_mesh(data.coronary.mesh_centering, color="lightgray", opacity=0.5)
        plotter_centering.add_mesh(data.coronary.get_coronary_central_line("centering"), color="red", line_width=5)
        plotter_centering.add_mesh(mesh_centering_from_label, opacity=0.5)
        plotter_centering.show_bounds()  #type: ignore
        plotter_centering.write_frame()

    plotter_original.close()
    plotter_centering.close()
    
    assert all(test_res_original), "The meshes generated from label and from coronary mesh should be close in original space. Please check the results carefully."
    assert all(test_res_centering), "The meshes generated from label and from coronary mesh should be close in centering space. Please check the results carefully."
    
    def uniform(t: torch.Tensor) -> torch.Tensor:
        t = (t - t.min()) / (t.max() - t.min())
        return torch.clamp(t * 255, 0, 255).to(torch.uint8)
    
    frames_w = uniform(torch.stack(frames_w))
    frames_h = uniform(torch.stack(frames_h))
    frames_d = uniform(torch.stack(frames_d))
    
    save_gif(output_dir / "frames_w.gif", frames_w, cmap='gray')
    save_gif(output_dir / "frames_h.gif", frames_h, cmap='gray')
    save_gif(output_dir / "frames_d.gif", frames_d, cmap='gray')  

    
    phase_0_data = reader.get_phase_0_data("LCA")
    central_line = phase_0_data.coronary.get_coronary_central_line("centering")
    mesh = phase_0_data.coronary.mesh_original
    
    assert len(central_line.shape) == 2 and central_line.shape[-1] == 3
    x_max, y_max, z_max = central_line.max(axis=0)
    x_min, y_min, z_min = central_line.min(axis=0)
    
    x_max_mesh, y_max_mesh, z_max_mesh = (mesh.points + 1).max(axis=0)  # plus/minus 1 to avoid small bias
    x_min_mesh, y_min_mesh, z_min_mesh = (mesh.points - 1).min(axis=0)

    if not (x_max < x_max_mesh and y_max < y_max_mesh and z_max < z_max_mesh 
            and x_min > x_min_mesh and y_min > y_min_mesh and z_min > z_min_mesh):
        print(f"warning: central line is out of mesh bounds, which is unexpected. This may indicate a problem in the coronary centering or the coronary mesh. Please check the results carefully.")
        print(f"Central line max: {(x_max, y_max, z_max)}, min: {(x_min, y_min, z_min)}")
        print(f"Mesh points max: {(x_max_mesh, y_max_mesh, z_max_mesh)}, min: {(x_min_mesh, y_min_mesh, z_min_mesh)}")

def test_volumes_reader(
    output_root_dir: Path,
    simulator_name: SimulatorName = SimulatorName.IDENTITY_CONTRAST,
):
    reader_name = ReaderName.VOLUMES_READER
    output_dir = output_root_dir / "readers_output" / str(reader_name)
    _, reader = reader_factory(simulator_name, reader_name)
    
    _read_and_save(reader, output_dir)

def test_volume_dvf_reader(
    output_root_dir: Path,
    simulator_name: SimulatorName = SimulatorName.IDENTITY_CONTRAST,
):
    reader_name = ReaderName.VOLUME_DVF_READER
    output_dir = output_root_dir / "readers_output" / str(reader_name)
    _, reader = reader_factory(simulator_name, reader_name)
    
    _read_and_save(reader, output_dir)

def test_rbf_reader(
    output_root_dir: Path,
    simulator_name: SimulatorName = SimulatorName.IDENTITY_CONTRAST,
):
    reader_name = ReaderName.RBF_READER
    _, reader = reader_factory(simulator_name, reader_name)

    for coronary_type in ["LCA", "RCA"]:
        output_dir = output_root_dir / "readers_output" / f"{reader_name}_{coronary_type}"
        _read_and_save(reader, output_dir, coronary_type=cast(Literal["LCA", "RCA"], coronary_type))

@pytest.mark.parametrize("coronary_type", ["LCA", "RCA"])
def test_rbf_reader_with_artifacts(
    output_root_dir: Path,
    volume_dvf_dir: Path,
    coronary_type: Literal["LCA", "RCA"],
    artifact_manager: ArtifactManager
):
    """RBFReader with artifact_manager: animate coronary + artifacts motion."""
    reader = RBFReader(
        volume_nii=volume_dvf_dir / "image.nii.gz",
        cavity_nii=volume_dvf_dir / "cavity.nii.gz",
        coronary_nii=volume_dvf_dir / "coronary.nii.gz",
        contrast_simulator=IdentityContrast(),
        artifact_manager=artifact_manager,
    )

    output_dir = output_root_dir / "readers_output" / f"rbf_reader_artifacts_{coronary_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    F = 15

    # ── Original-coordinate animation ──
    plotter_original = pv.Plotter(off_screen=True)
    plotter_original.open_gif(str(output_dir / "animation_artifacts_original.gif"))

    # ── Centering-coordinate animation ──
    plotter_centering = pv.Plotter(off_screen=True)
    plotter_centering.open_gif(str(output_dir / "animation_artifacts_centering.gif"))

    for phase_idx in range(F):
        phase = CardiacPhase.from_index(phase_idx, F)
        data = reader.get_dynamic_data(phase, coronary_type)

        # --- original view ---
        plotter_original.clear()
        plotter_original.add_mesh(data.coronary.mesh_original, color="lightgray", opacity=0.5)
        if data.coronary.artifacts_original is not None:
            for art in data.coronary.artifacts_original:
                plotter_original.add_mesh(art.mesh, color="red" if "marker" in art.name else "orange", opacity=0.9)
        plotter_original.show_bounds()  # type: ignore
        plotter_original.write_frame()

        # --- centering view ---
        plotter_centering.clear()
        plotter_centering.add_mesh(data.coronary.mesh_centering, color="lightgray", opacity=0.5)
        if data.coronary.artifacts_centering is not None:
            for art in data.coronary.artifacts_centering:
                plotter_centering.add_mesh(art.mesh, color="red" if "marker" in art.name else "orange", opacity=0.9)
        plotter_centering.show_bounds()  # type: ignore
        plotter_centering.write_frame()

    plotter_original.close()
    plotter_centering.close()

    # Verify artifacts were generated and stored
    cor = reader.origin_data[CoronaryType[coronary_type]]
    assert cor.artifacts_original is not None
    assert len(cor.artifacts_original) == 4, f"Expected 4 artifacts, got {len(cor.artifacts_original)}"
    names = {a.name for a in cor.artifacts_original}
    assert "guide_wire" in names
    assert any(n.startswith("marker_") for n in names)


def test_rbf_reader_static_data_chest(
    artifact_manager: ArtifactManager,
    output_root_dir: Path,
    simulator_name: SimulatorName = SimulatorName.STATIC_IODINE_CONTRAST,
    data_with_background_dir: Path = TEST_DATA_ROOT_DIR / "chest_data",
    name: str = "chest_static_data"
):
    """
    Save get_static_data() layers for chest CTA + background CT.
    """
    cleanup_gpu_memory()
    chest_dir = data_with_background_dir
    assert chest_dir.exists(), f"chest_data not found: {chest_dir}"

    reader = RBFReader(
        volume_nii=chest_dir / "image.nii.gz",
        cavity_nii=chest_dir / "cavity.nii.gz",
        coronary_nii=chest_dir / "coronary.nii.gz",
        contrast_simulator=simulator_factory(simulator_name),
        background_nii=chest_dir / "background.nii.gz",
        artifact_manager=artifact_manager,
    )

    output_dir = output_root_dir / "readers_output" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = reader.get_static_data("LCA")
    assert len(layers) >= 1, "get_static_data should return at least 1 layer"

    def _save_slices(vol: torch.Tensor, save_dir: Path, prefix: str = "") -> None:
        """Save middle slices along W, H, D axes as PNG."""
        vol_sq = vol.squeeze()  # (W, H, D)
        W, H, D = vol_sq.shape

        def _normalize(slice_2d: torch.Tensor) -> torch.Tensor:
            lo, hi = slice_2d.min(), slice_2d.max()
            if hi > lo:
                return ((slice_2d - lo) / (hi - lo) * 255).to(torch.uint8)
            return torch.zeros_like(slice_2d, dtype=torch.uint8)

        for axis_name, slice_2d in [
            ("w", vol_sq[W // 2, :, :]),       # sagittal,   shape (H, D)
            ("h", vol_sq[:, H // 2, :]),       # coronal,    shape (W, D)
            ("d", vol_sq[:, :, D // 2]),       # axial,      shape (W, H)
        ]:
            p = save_dir / f"{prefix}slice_{axis_name}.png"
            save_png(p, _normalize(slice_2d))

    for i, img in enumerate(layers):
        layer_dir = output_dir / f"layer_{i}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Save volume as NIfTI
        save_nii(layer_dir / "volume.nii.gz", img.data, img.affine)

        # Save middle slices along three axes
        _save_slices(img.data, layer_dir)

    # Also save the dynamic ROI from get_data for reference
    dynamic_data = reader.get_dynamic_data(CardiacPhase(0), "LCA")
    save_nii(output_dir / "dynamic_volume_phase0.nii.gz",
             dynamic_data.coronary.volume.data,
             dynamic_data.coronary.centering_affine)
    _save_slices(dynamic_data.coronary.volume.data, output_dir, prefix="dynamic_")

    combined_volume = layers[0].data.squeeze()
    combined_affine = layers[0].affine
    _save_slices(combined_volume, output_dir, prefix=f"combined_layer_0_")

    for i in range(1, len(layers)):
        img = layers[i]
        combined_volume += resample(
            img.data.squeeze(),
            img.affine,
            combined_affine,
            combined_volume.shape[-3:]  # (W, H, D)
        )
        _save_slices(combined_volume, output_dir, prefix=f"combined_layer_{i}_")
        

    combined_volume += resample(
        dynamic_data.coronary.volume.data.squeeze(),
        dynamic_data.coronary.centering_affine,
        combined_affine,
        combined_volume.shape[-3:]  # (W, H, D)
    ).squeeze()  # remove batch dim

    save_nii(output_dir / "combined_volume.nii.gz", combined_volume, combined_affine)
    _save_slices(combined_volume, output_dir, prefix="combined_")

    print(f"\nAll static layers saved to {output_dir}")
    print(f"Total static layers: {len(layers)}")
    for i, img in enumerate(layers):
        print(f"  Layer {i}: shape={img.data.shape}")

    # ── Visualize coronary mesh + artifacts in centering coordinate ──
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    mesh_centering_from_label = get_mesh_in_world(dynamic_data.coronary.label.data, dynamic_data.coronary.centering_affine, max_points=None)

    for _ in range(3):  # write a few frames for visibility
        plotter.clear()
        plotter.add_mesh(dynamic_data.coronary.mesh_centering, color="lightgray", opacity=0.5)
        plotter.add_mesh(mesh_centering_from_label, opacity=0.5, color="blue")
        if dynamic_data.coronary.artifacts_centering is not None:
            for art in dynamic_data.coronary.artifacts_centering:
                plotter.add_mesh(art.mesh, color="red" if "marker" in art.name else "orange", opacity=0.9)
        plotter.show_bounds()  # type: ignore

    plotter.screenshot(str(output_dir / "centering_coronary_artifacts.png"))
    plotter.close()
    print(f"Saved centering visualization to {output_dir}")


def test_rbf_reader_static_data_chest_different(
    artifact_manager: ArtifactManager,
    output_root_dir: Path,
    simulator_name: SimulatorName = SimulatorName.HU2MU_CONTRAST,
    data_with_background_dir: Path = TEST_DATA_ROOT_DIR / "chest_data" / "cta_single",
    name="chest_static_data_different"
):
    """
    use chest data from different source, to verify the static data reader works for different data.
    """
    test_rbf_reader_static_data_chest(
        
    artifact_manager, output_root_dir, simulator_name, data_with_background_dir, name
    )


def resample(
    data: torch.Tensor,
    old_affine: np.ndarray,
    new_affine: np.ndarray,
    new_shape: tuple[int, ...]
) -> torch.Tensor:
    """
    Resample data from old_affine space to new_affine space using grid_sample.

    坐标系约定:
        - data shape: (..., W, H, D) — 最后三维为空间维度
        - affine: 4x4 矩阵, voxel (x,y,z) = (w,h,d) → world
        - new_shape: (W_new, H_new, D_new) — 输出体积的形状

    原理:
        对新空间的每个体素 (w,h,d):
            old_voxel = inv(old_affine) @ new_affine @ (w,h,d,1)
        然后用 F.grid_sample 在 old 空间中插值。

    正确处理:
        - Affine 中的反射 (负缩放)
        - 旋转、剪切
        - 超出边界的体素 → 按 padding_mode="zeros" 处理
    """
    import torch.nn.functional as F

    device = data.device
    dtype = data.dtype

    W_new, H_new, D_new = new_shape
    old_shape = data.shape[-3:]  # (W_old, H_old, D_old)

    # ── 1. 构造新空间网格 ──
    gw, gh, gd = torch.meshgrid(
        torch.arange(W_new, device=device, dtype=torch.float32),
        torch.arange(H_new, device=device, dtype=torch.float32),
        torch.arange(D_new, device=device, dtype=torch.float32),
        indexing="ij",
    )  # each (W, H, D)

    # ── 2. 坐标变换: new_voxel → world → old_voxel ──
    old_affine_t = torch.from_numpy(old_affine).to(device, torch.float32)
    new_affine_t = torch.from_numpy(new_affine).to(device, torch.float32)
    T = torch.linalg.inv(old_affine_t) @ new_affine_t  # (4, 4)

    ones = torch.ones_like(gw)
    new_hom = torch.stack([gw, gh, gd, ones], dim=-1)  # (W, H, D, 4)
    old_hom = torch.einsum("ij,...j->...i", T, new_hom)  # (W, H, D, 4)
    old_voxel = old_hom[..., :3]  # (W, H, D, 3), last dim = (w,h,d)

    # ── 3. 归一化到 [-1, 1] (grid_sample 要求) ──
    old_shape_t = torch.tensor(old_shape, device=device, dtype=torch.float32)
    grid_norm = old_voxel / (old_shape_t - 1) * 2 - 1  # (W, H, D, 3)

    # ── 4. 调整 grid 和 data 的轴序: grid_sample 3D 用 (N, D, H, W, 3) ──
    grid_norm = grid_norm.permute(2, 1, 0, 3).unsqueeze(0)  # (1, D, H, W, 3)

    # data: (..., W, H, D) → (1, 1, D, H, W)
    if data.dim() == 3:
        data_5d = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 5:
        data_5d = data
    else:
        raise ValueError(f"Unsupported data dim: {data.dim()}")
    data_permuted = data_5d.permute(0, 1, 4, 3, 2)  # (1, 1, D_old, H_old, W_old)

    # ── 5. grid_sample ──
    sampled = F.grid_sample(
        data_permuted,
        grid_norm,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (1, 1, D_new, H_new, W_new)

    # ── 6. 恢复轴序: (1, 1, W_new, H_new, D_new) ──
    sampled = sampled.permute(0, 1, 4, 3, 2)  # (1, 1, W, H, D)

    return sampled.squeeze().to(dtype)

if __name__ == "__main__":
    test_volume_dvf_reader(
        OUTPUT_ROOT_DIR
    )