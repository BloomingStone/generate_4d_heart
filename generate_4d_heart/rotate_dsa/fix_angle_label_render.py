from pathlib import Path
from typing import Sequence
import numpy as np
import json

import torch
from tqdm import tqdm

from .rotate_drr import CArmGeometry
from ..types import CoronaryType
from .torch3d_label_renderer import Torch3DLabelRenderer
from .data_reader import DataReader
from .cardiac_phase import CardiacPhase
from ..saver import save_pngs, save_gif as _save_gif


class Projector:
    def __init__(
        self, 
        reader: DataReader, 
        c_arm_geometry: CArmGeometry = CArmGeometry(), 
        device: torch.device = torch.device("cuda:0")
    ):
        self.c_arm_geometry = c_arm_geometry
        self.device = device
        self.renderer = Torch3DLabelRenderer(c_arm_geometry)
        self.reader = reader

    def project(
        self, 
        phase: CardiacPhase | list[CardiacPhase], 
        coronary_type: CoronaryType,
        angles_degree: Sequence[tuple[float, float]] | tuple[float, float] | np.ndarray,
    ) -> tuple[np.ndarray, list[dict]]:           
        if isinstance(angles_degree, tuple):
            angles_np = np.array([angles_degree])
        elif isinstance(angles_degree, list):
            angles_np = np.array(angles_degree, dtype=np.float32)
        else:
            angles_np = np.array(angles_degree, dtype=np.float32)
        
        if angles_np.ndim == 1:
            angles_np = angles_np[None]
        
        assert angles_np.ndim == 2, f"angles must be a 2D array of shape (N, 2) or (2,), but got shape {angles_np.shape}"
        N, D = angles_np.shape
        if D != 2:
            raise ValueError(f"angles_degree must have shape (N, 2) or (2,), but got {angles_np.shape}")
        
        angles_np = np.array(angles_degree, dtype=np.float32)
        angles_rad = np.radians(angles_np)
        alpha_rad = torch.from_numpy(angles_rad[:, 0]).to(self.device)
        beta_rad = torch.from_numpy(angles_rad[:, 1]).to(self.device)
        
        if not isinstance(phase, list):
            frames_meta = [{
                "phase": float(phase),
                "primary_angle": float(angle[0]),
                "secondary_angle": float(angle[1]),
            } for angle in angles_np]
            print(f"Projecting for phase {phase} and coronary type {coronary_type} at angles {angles_degree}")
            data = self.reader.get_dynamic_data(phase, coronary_type)
            mesh = data.coronary.mesh_centering

            return self.renderer.render_silhouette_depth_by_angles(mesh, alpha_rad, beta_rad)[0], frames_meta
    
        assert len(phase) == N, f"Length of phase list ({len(phase)}) must match number of angles ({N})"
        
        all_labels = []
        frames_meta = []
        for i in tqdm(range(N), desc="Projecting frames"):
            data = self.reader.get_dynamic_data(phase[i], coronary_type)
            mesh = data.coronary.mesh_centering
            label_2d, _ = self.renderer.render_silhouette_depth_by_angles(mesh, alpha_rad[i:i+1], beta_rad[i:i+1])
            all_labels.append(label_2d[0])
            frames_meta.append({
                "phase": float(phase[i]),
                "primary_angle": float(angles_np[i][0]),
                "secondary_angle": float(angles_np[i][1]),
            })
        return np.stack(all_labels, axis=0), frames_meta

    
    def project_and_save(
        self, 
        phase: CardiacPhase | list[CardiacPhase], 
        coronary_type: CoronaryType,
        angles_degree: Sequence[tuple[float, float]] | tuple[float, float] | np.ndarray,
        save_dir: Path,
        save_gif: bool = False,
    ):
        label_2d, frames_meta = self.project(phase, coronary_type, angles_degree)
        save_pngs(save_dir/"labels", (label_2d*255).astype(np.uint8))
        phase_0_data = self.reader.get_phase_0_data(coronary_type)
        phase_0_meta = phase_0_data.save(
            output_dir=save_dir,
            save_types=["cor-mesh", "coronary"],
        )
        
        if save_gif:
            _save_gif(save_dir/"labels.gif", (label_2d*255).astype(np.uint8), fps=30, cmap='gray')
        
        with open(save_dir/"meta.json", "w") as f:
            json.dump({
                "c_arm_geometry": self.c_arm_geometry.to_dict(),
                "phase_0_meta": phase_0_meta,
                "frames_meta": frames_meta,
            }, f, indent=4)