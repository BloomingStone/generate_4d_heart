from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json
import math

from tqdm import tqdm
import pyvista as pv
import numpy as np

from .data_reader import DataReader
from .contrast_simulator import ContrastSimulator
from .rotate_drr import RotateDRR, CArmGeometry
from .types import Sec
from .cardiac_phase import CardiacPhase
from ..saver import save_tif, save_gif, save_pngs, save_deepthmap_gif


class LabelPlotter:
    def __init__(self, c_arm: CArmGeometry):
        self.view_angle = 2 * math.atan(c_arm.width * c_arm.delx / 2 / c_arm.sdd) / math.pi * 180
        self.width = c_arm.width
        self.height = c_arm.height
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.window_size = (self.width, self.height)
        self.plotter.remove_all_lights()
        self.plotter.background_color = pv.Color("black")
        self.plotter.camera.view_angle = self.view_angle
    
    def get_label_and_depth_map(
        self,
        mesh: pv.PolyData,
        camera_pos: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.plotter.add_mesh(mesh, color='white')    # white mesh as foreground
        self.plotter.camera.position = camera_pos
        self.plotter.show(auto_close=False)
        deepth_map = self.plotter.get_image_depth()
        label = self.plotter.screenshot(return_img=True)
        self.plotter.clear()
        
        assert label is not None
        label = (label[:, :, 0] > 0).astype(np.uint8)
        return label, deepth_map


@dataclass
class PhysicalConfig:
    cardiac_cycle_time: Sec = 1.0   # Cardiac cycle time
    # can add more config like breath cycle time etc..


@dataclass
class RotateDSA:
    reader: DataReader
    constrast_sim: ContrastSimulator
    drr: RotateDRR
    physical_config: PhysicalConfig = field(default_factory=PhysicalConfig)
    label_plotter: LabelPlotter = field(init=False)
    
    def __post_init__(self):
        self.label_plotter = LabelPlotter(self.drr.c_arm_cfg)

    def run(
        self, 
        coronary_type: Literal["LCA", "RCA"] = "LCA",
        gray_reverse: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run whole process of rotate DRR
        Returns:
            np.ndarray: DRR images shape = (t, w, h), t = rotate_parameters.total_frame
            np.ndarray: coroanry labels, shape is the same as DRR images, 0: background, 1: coronary
            np.ndarray: depth masks, shape is the same as DRR images, nan for background, depth value for coronary
        """
        assert coronary_type in ["LCA", "RCA"]
        total_frame = self.drr.rotate_cfg.total_frame
        w, h = self.drr.image_size
        frames = np.zeros((total_frame, w, h))
        labels = np.zeros((total_frame, w, h), dtype=np.uint8)
        depth_maps = np.zeros((total_frame, w, h), dtype=np.float32)
        for f in tqdm(range(total_frame), desc="Generating Rotate DSA..."):
            phase = self._get_phase_at_frame(f)
            read_res = self.reader.get_data(phase, coronary_type).to_device(self.drr.device)
            coronary_label = read_res.coronary.label
            affine = read_res.coronary.centering_affine
            
            volume = self.constrast_sim.simulate(
                ori_volume=read_res.volume,
                cavity_label=read_res.cavity_label,
                coronary_label=coronary_label
            )
            drr_res = self.drr.get_projection_at_frame(
                frame=f,
                volume=volume,
                coronary=coronary_label,
                affine=affine
            )
            frames[f] = drr_res.cpu().numpy()
            
            _, T = self.drr.get_R_T_at_frame(f)
            label, depth_map = self.label_plotter.get_label_and_depth_map(
                mesh=read_res.coronary.mesh_in_world,
                camera_pos=T.squeeze().cpu().numpy().tolist()
            )
            labels[f] = label
            depth_maps[f] = depth_map
        
        frames = ((frames - frames.min()) / (frames.max() - frames.min()))*255
        frames = frames.astype(np.uint8)
        if gray_reverse:
            frames = - frames + 255

        return frames, labels, depth_maps


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
            
            _, T = self.drr.get_R_T_at_frame(f)
            label, depth_map = self.label_plotter.get_label_and_depth_map(
                mesh=read_res.coronary.mesh_in_world,
                camera_pos=T.squeeze().cpu().numpy().tolist()
            )
            labels[f] = label
            depth_maps[f] = depth_map
        
        return labels, depth_maps
    
    
    def run_and_save(
        self, 
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"],
        gray_reverse: bool = True,
        gif_fps: int = 30   # gif may not support too high fps (like fps=60 may cause gif slow)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames, labels, depth_maps = self.run(coronary_type, gray_reverse)
        save_tif(output_dir / "rotate_dsa.tif", frames)
        save_tif(output_dir / "label.tif", labels)
        
        save_gif(output_dir / "rotate_dsa.gif", frames, gif_fps)
        save_gif(output_dir / "label.gif", labels*255, gif_fps)
        save_deepthmap_gif(output_dir / "depth_map.gif", depth_maps, gif_fps)
        
        save_pngs(output_dir / "rotate_dsa", frames)
        save_pngs(output_dir / "label", labels*255)
        
        np.save(output_dir / "label.npy", labels)
        np.save(output_dir / "depth_map.npy", depth_maps)
        
        with open(output_dir / "rotate_dsa.json", "w") as f:
            json.dump(self.get_geometry_json(coronary_type), f)
        
        return frames, labels, depth_maps
    
    
    def run_and_save_no_drr(
        self, 
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"],
        gif_fps: int = 30   # gif may not support too high fps (like fps=60 may cause gif slow)
    ) -> tuple[np.ndarray, np.ndarray]:
        labels, depth_maps = self.run_no_drr(coronary_type)
        save_tif(output_dir / "label.tif", labels)
        
        save_gif(output_dir / "label.gif", labels*255, gif_fps)
        save_deepthmap_gif(output_dir / "depth_map.tif", depth_maps, gif_fps)
        
        save_pngs(output_dir / "label", labels*255)
        
        np.save(output_dir / "label.npy", labels)
        np.save(output_dir / "depth_map.npy", depth_maps)
        
        with open(output_dir / "rotate_dsa.json", "w") as f:
            json.dump(self.get_geometry_json(coronary_type), f)
        
        return labels, depth_maps
    
    
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