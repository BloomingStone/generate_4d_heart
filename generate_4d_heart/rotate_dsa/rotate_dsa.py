from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any
import json
import math

from tqdm import tqdm
import pyvista as pv
import numpy as np
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)
from pytorch3d.structures import Meshes

from .data_reader import DataReader
from .contrast_simulator import ContrastSimulator
from .rotate_drr import RotateDRR, CArmGeometry
from .types import Sec
from .cardiac_phase import CardiacPhase
from ..saver import save_tif, save_gif, save_pngs, save_deepthmap_gif
from .postprocess import postprocess_drr


class Torch3DLabelRenderer:
    def __init__(self, c_arm: CArmGeometry):
        self.fov = 2 * math.atan(c_arm.width * c_arm.delx / 2 / c_arm.sdd) / math.pi * 180
        self.width = int(c_arm.width)
        self.height = int(c_arm.height)
        self.zfar = c_arm.sdd*1.2
        self.raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.0,
            faces_per_pixel=1
        )
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            raise RuntimeError("No CUDA device available for Torch3DLabelRenderer")

    @torch.no_grad()
    def render(self, mesh_pv: pv.PolyData, R: torch.Tensor, T: torch.Tensor):
        verts = torch.from_numpy(np.array(mesh_pv.points)).float()
        faces_np = np.array(mesh_pv.faces.reshape(-1, 4)[:, 1:])  # skip the first number which is the number of points per face
        faces = torch.from_numpy(faces_np).long()
        mesh = Meshes([verts.to(self.device)], [faces.to(self.device)])

        
        R = R.squeeze()
        T = T.squeeze()
        cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=self.fov,
            R=R[None].to(self.device),      # pytorch3D 默认使用行向量，因此这里输入的 作用于列向量的 R_c2w 可以不用转置，直接视为R_w2c 矩阵
            T=(-R.T @ T)[None].to(self.device), 
            zfar=self.zfar
        )
        
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )
        
        # rasterize
        fragments = rasterizer(mesh)
        depth = fragments.zbuf[0, ..., 0]  # depth buffer
        # silhouette
        silhouette = (fragments.pix_to_face[..., 0] >= 0).float()
        return silhouette.cpu().numpy(), depth.cpu().numpy()

@dataclass
class PhysicalConfig:
    cardiac_cycle_time: Sec = 1.0   # Cardiac cycle time
    # can add more config like breath cycle time etc..


@dataclass
class RotateDsaOutput:
    raw_frames: np.ndarray
    vis_frames: np.ndarray
    labels: np.ndarray
    depth_maps: np.ndarray
    meta: dict[str, Any]

@dataclass
class RotateDSA:
    reader: DataReader
    drr: RotateDRR
    physical_config: PhysicalConfig = field(default_factory=PhysicalConfig)
    label_plotter: Torch3DLabelRenderer = field(init=False)
    
    def __post_init__(self):
        self.label_plotter = Torch3DLabelRenderer(self.drr.c_arm_cfg)

    def run(
        self, 
        coronary_type: Literal["LCA", "RCA"] = "LCA",
    ) -> RotateDsaOutput:
        """
        Run whole process of rotate DRR
        Returns:
            RotateDsaOutput: Contains all generated outputs and metadata
        """
        assert coronary_type in ["LCA", "RCA"]
        total_frame = self.drr.rotate_cfg.total_frame
        w, h = self.drr.image_size
        frames = torch.zeros(total_frame, w, h).to(torch.float32)
        labels = np.zeros((total_frame, w, h), dtype=np.uint8)
        depth_maps = np.zeros((total_frame, w, h), dtype=np.float32)
        for f in tqdm(range(total_frame), desc="Generating Rotate DSA..."):
            phase = self._get_phase_at_frame(f)
            time = f / self.drr.rotate_cfg.fps
            read_res = self.reader.get_data(phase, coronary_type, time).to_device(self.drr.device)
            coronary = read_res.coronary
            
            drr_res = self.drr.get_projection_at_frame(
                frame=f,
                volume=coronary.volume,
                coronary=coronary.label,
                affine=coronary.centering_affine
            )
            if frames.device != drr_res.device:
                frames = frames.to(drr_res.device)
            frames[f] = drr_res
            
            label, depth_map = self.label_plotter.render(
                coronary.mesh_centering,
                *self.drr.get_R_T_at_frame(f)
            )
            labels[f] = label
            depth_maps[f] = depth_map
        
        # Beer-Lambert: line integral (I = \int mu dx) → intensity. (I = I0 * exp(-\int mu dx))
        frames = torch.exp( - frames)
        vis_frames, post_process_meta = postprocess_drr(frames)

        return RotateDsaOutput(
            raw_frames=frames.cpu().numpy().astype(np.float32),
            vis_frames=vis_frames.cpu().numpy().astype(np.uint8),
            labels=labels,
            depth_maps=depth_maps,
            meta=post_process_meta
        )


    def run_no_drr(
        self, 
        coronary_type: Literal["LCA", "RCA"] = "LCA"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run whole process of rotate DRR
        Returns:
            np.ndarray: coroanry labels, shape is the same as DRR images, 0: background, 1: coronary
            np.ndarray: depth masks, shape is the same as DRR images, nan for background, depth value for coronary
        """
        assert coronary_type in ["LCA", "RCA"]
        total_frame = self.drr.rotate_cfg.total_frame
        w, h = self.drr.image_size
        labels = np.zeros((total_frame, w, h), dtype=np.uint8)
        depth_maps = np.zeros((total_frame, w, h), dtype=np.float32)
        for f in tqdm(range(total_frame), desc="Generating Rotate DSA without DRR..."):
            phase = self._get_phase_at_frame(f)
            read_res = self.reader.get_data(phase, coronary_type).to_device(self.drr.device)
            
            label, depth_map = label, depth_map = self.label_plotter.render(
                read_res.coronary.mesh_centering,
                *self.drr.get_R_T_at_frame(f)
            )
            labels[f] = label
            depth_maps[f] = depth_map
        
        return labels, depth_maps
    
    
    def run_and_save(
        self, 
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"],
        gif_fps: int = 30   # gif may not support too high fps (like fps=60 may cause gif slow)
    ) -> RotateDsaOutput:
        output = self.run(coronary_type)
        save_tif(output_dir / "rotate_dsa_raw.tif", output.raw_frames)
        
        save_gif(output_dir / "rotate_dsa.gif", output.vis_frames, gif_fps,cmap='gray', vmin=0, vmax=255)
        save_gif(output_dir / "rotate_dsa_raw.gif", output.raw_frames, gif_fps,cmap='gray')
        save_gif(output_dir / "label.gif", output.labels*255, gif_fps,cmap='gray')
        save_deepthmap_gif(output_dir / "depth_map.gif", output.depth_maps, gif_fps)
        
        save_pngs(output_dir / "rotate_dsa", output.vis_frames)
        save_pngs(output_dir / "label", output.labels*255)
        
        np.savez_compressed(output_dir / "label.npz", output.labels)
        np.savez_compressed(output_dir / "depth_map.npz", output.depth_maps)
        
        phase_0_data = self.reader.get_phase_0_data(coronary_type)
        meta = phase_0_data.save(
            output_dir, 
            save_nii_type=["coronary"], save_mesh=True, save_central_line=True,
            coordinate_system="coronary_centering"
        )
        
        json_data = self.get_geometry_json(coronary_type)
        json_data["postprocess_meta"] = output.meta
        json_data["phase_0_save_meta"] = meta
        with open(output_dir / "rotate_dsa.json", "w") as f:
            json.dump(json_data, f, indent=2)
        
        return output
    
    
    def run_and_save_no_drr(
        self, 
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"],
        gif_fps: int = 30   # gif may not support too high fps (like fps=60 may cause gif slow)
    ) -> tuple[np.ndarray, np.ndarray]:
        labels, depth_maps = self.run_no_drr(coronary_type)
        
        save_gif(output_dir / "label.gif", labels*255, gif_fps, cmap='gray')
        save_deepthmap_gif(output_dir / "depth_map.gif", depth_maps, gif_fps)
        
        save_pngs(output_dir / "label", labels*255)
        
        np.savez_compressed(output_dir / "label.npz", labels)
        np.savez_compressed(output_dir / "depth_map.npz", depth_maps)
        
        phase_0_data = self.reader.get_phase_0_data(coronary_type)
        meta = phase_0_data.save(
            output_dir, 
            save_nii_type=["coronary"], save_mesh=True, save_central_line=True,
            coordinate_system="coronary_centering"
        )
        
        json_data = self.get_geometry_json(coronary_type)
        json_data["phase_0_save_meta"] = meta
        with open(output_dir / "rotate_dsa.json", "w") as f:
            json.dump(json_data, f, indent=2)
        
        return labels, depth_maps
    
    def save_no_run(
        self, 
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"],
    ) -> None:
        phase_0_data = self.reader.get_phase_0_data(coronary_type)
        meta = phase_0_data.save(
            output_dir, 
            save_nii_type=["coronary"], save_mesh=True, save_central_line=True,
            coordinate_system="coronary_centering"
        )
        
        json_data = self.get_geometry_json(coronary_type)
        json_data["phase_0_save_meta"] = meta
        with open(output_dir / "rotate_dsa.json", "w") as f:
            json.dump(json_data, f, indent=2)
    
    
    def _get_phase_at_frame(self, frame: int) -> CardiacPhase:
        return CardiacPhase.from_time(
            frame / self.drr.rotate_cfg.fps, 
            self.physical_config.cardiac_cycle_time
        )
    
    def get_geometry_json(self, coronary_type: Literal["LCA", "RCA"]) -> dict:
        res = {}
        res["coronary_type"] = coronary_type
        res["volume_size"] = self.reader.volume_size
        res["volume_affine"] = self.reader.volume_affine.tolist()
        res["lca_centering_affine"] = self.reader.lca_centering_affine.tolist()
        res["rca_centering_affine"] = self.reader.rca_centering_affine.tolist()
        res["c_arm_geometry"] = self.drr.c_arm_cfg.to_dict()
        res["rotate_parameters"] = self.drr.rotate_cfg.to_dict()
        if (additional_config := self.drr.get_additional_config()):
            res["additional_config"] = additional_config
        res["frames"] = []
        
        for f in range(self.drr.rotate_cfg.total_frame):
            alpha, beta, _ = self.drr.rotate_cfg.get_rotation_angle_at_frame(f)
            d = {
                "time_s": f / self.drr.rotate_cfg.fps,
                "frame": f,
                "phase": float(self._get_phase_at_frame(f)),
                "alpha_degree": alpha,
                "beta_degree": beta,
            }
            
            res["frames"].append(d)
        
        return res