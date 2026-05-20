from typing import Literal, cast
from pathlib import Path

import pyvista as pv
import pytest
import torch
import numpy as np
from tqdm import tqdm

from generate_4d_heart.rotate_dsa.data_reader.data_reader import DataReader, get_mesh_in_world
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.saver import save_gif

from utils import reader_factory, ReaderName, SimulatorName, OUTPUT_ROOT_DIR


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
    frames_w = torch.rand(F, H, D)
    frames_h = torch.rand(F, W, D)
    frames_d = torch.rand(F, W, H)

    volume = None
    
    plotter_original = pv.Plotter(off_screen=True)
    plotter_original.open_gif(output_dir / "animation_original.gif")
    
    plotter_centering = pv.Plotter(off_screen=True)
    plotter_centering.open_gif(output_dir / "animation_centering.gif")
    
    test_res_original = []
    test_res_centering = []
    for phase_idx in tqdm(range(F), desc=f"Generating {F} continous frames in one cardiac cycle"):
        phase = CardiacPhase.from_index(phase_idx, F)

        data = reader.get_data(phase, coronary_type)
        volume = data.coronary.volume[0, 0]
        
        frames_w[phase_idx] = volume[W//2]
        frames_h[phase_idx] = volume[:, H//2]
        frames_d[phase_idx] = volume[:, :, D//2]
        
        mesh_original_from_label = get_mesh_in_world(data.coronary.label, data.coronary.original_affine, max_points=None)
        test_res_original.append(np.allclose(mesh_original_from_label.bounds, data.coronary.mesh_original.bounds, atol=1, rtol=0.1))
        
        mesh_centering_from_label = get_mesh_in_world(data.coronary.label, data.coronary.centering_affine, max_points=None)
        test_res_centering.append(np.allclose(mesh_centering_from_label.bounds, data.coronary.mesh_centering.bounds, atol=1, rtol=0.1))
        
        plotter_original.clear()
        plotter_original.add_mesh(data.coronary.mesh_original, color="lightgray", opacity=0.5)
        plotter_original.add_mesh(mesh_original_from_label, opacity=0.5)
        plotter_original.show_bounds()  #type: ignore
        plotter_original.write_frame()
        
        plotter_centering.clear()
        plotter_centering.add_mesh(data.coronary.mesh_centering, color="lightgray", opacity=0.5)
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
    
    frames_w = uniform(frames_w)
    frames_h = uniform(frames_h)
    frames_d = uniform(frames_d)
    
    save_gif(output_dir / "frames_w.gif", frames_w, cmap='gray')
    save_gif(output_dir / "frames_h.gif", frames_h, cmap='gray')
    save_gif(output_dir / "frames_d.gif", frames_d, cmap='gray')  

    
    phase_0_data = reader.get_phase_0_data("LCA")
    central_line = phase_0_data.coronary.get_coronary_central_line("coronary_centering")
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
    output_dir = output_root_dir / "readers_output" / str(reader_name)
    _, reader = reader_factory(simulator_name, reader_name)

    _read_and_save(reader, output_dir)


if __name__ == "__main__":
    test_volume_dvf_reader(
        OUTPUT_ROOT_DIR
    )