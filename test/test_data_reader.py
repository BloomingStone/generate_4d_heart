from typing import Literal
from pathlib import Path
from functools import lru_cache

import numpy as np
import pyvista as pv
import pytest
import torch
from tqdm import tqdm

from generate_4d_heart.rotate_dsa.data_reader import (
    VolumesReader, DataReader, VolumeDVFReader,
    CoronaryBoundLVLinearEnhancer, RBFReader
)
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.saver import save_gif
from utils import output_root_dir, test_data_root_dir


def _read_and_save(reader: DataReader, output_dir: Path, coronary_type: Literal["LCA", "RCA"] = "LCA") -> None:
    # delete result in output_dir first
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

    mesh_list = []
    for phase_idx in tqdm(range(F), desc=f"Generating {F} continous frames in one cardiac cycle"):
        phase = CardiacPhase.from_index(phase_idx, F)

        data = reader.get_data(phase, coronary_type)
        volume = data.volume[0, 0]
        
        frames_w[phase_idx] = volume[W//2]
        frames_h[phase_idx] = volume[:, H//2]
        frames_d[phase_idx] = volume[:, :, D//2]
        
        mesh_list.append(data.coronary.mesh_in_world)
    
    def uniform(t: torch.Tensor) -> torch.Tensor:
        t = (t - t.min()) / (t.max() - t.min())
        return torch.clamp(t * 255, 0, 255).to(torch.uint8)
    
    frames_w = uniform(frames_w)
    frames_h = uniform(frames_h)
    frames_d = uniform(frames_d)
    
    save_gif(output_dir / "frames_w.gif", frames_w, cmap='gray')
    save_gif(output_dir / "frames_h.gif", frames_h, cmap='gray')
    save_gif(output_dir / "frames_d.gif", frames_d, cmap='gray')
    
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(output_dir / "animation.gif")

    for mesh in mesh_list:
        plotter.clear()
        plotter.add_mesh(mesh)
        plotter.write_frame()

    plotter.close()
    
    phase_0_data = reader.get_phase_0_data("LCA")
    central_line = phase_0_data.get_coronary_central_line("coroanry_centering")
    mesh = phase_0_data.coronary.mesh_in_world
    
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

def test_volumes_reader():
    volumes_dir = test_data_root_dir / "volumes"
    assert volumes_dir.exists()
    output_dir = output_root_dir / "readers_output" / "volumes"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    reader = VolumesReader(
        image_dir=volumes_dir / "image",
        cavity_dir=volumes_dir / "cavity",
        coronary_dir=volumes_dir / "coronary",
    )
    
    _read_and_save(reader, output_dir)

def test_volume_dvf_reader():
    data_dir = test_data_root_dir / "volume_with_dvf"
    assert data_dir.exists()
    output_dir = output_root_dir / "readers_output" / "dvf"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    reader = VolumeDVFReader(
        image_nii=data_dir / "image.nii.gz",
        cavity_nii=data_dir / "cavity.nii.gz",
        coronary_nii=data_dir / "coronary.nii.gz",
        dvf_dir=data_dir / "dvf",
        roi_json=data_dir / "Normal_01.json",
        movement_enhancer=CoronaryBoundLVLinearEnhancer(enhance_coronary="LCA")
    )
    
    _read_and_save(reader, output_dir)

def test_rbf_reader():
    data_dir = test_data_root_dir / "volume_with_dvf"
    assert data_dir.exists()
    output_dir = output_root_dir / "readers_output" / "rbf"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    reader = RBFReader(
        image_nii=data_dir / "image.nii.gz",
        cavity_nii=data_dir / "cavity.nii.gz",
        coronary_nii=data_dir / "coronary.nii.gz"
    )
    
    _read_and_save(reader, output_dir, coronary_type="RCA")


if __name__ == "__main__":
    # test_volumes_reader()
    # test_volume_dvf_reader()
    test_rbf_reader()