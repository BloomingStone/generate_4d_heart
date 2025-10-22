from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json

import torch
from tqdm import tqdm

from .data_reader import DataReader
from .contrast_simulator import ContrastSimulator
from .rotate_drr import RotateDRR
from .types import Sec
from .cardiac_phase import CardiacPhase
from ..saver import save_tif, save_gif, save_pngs


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

    def run(
        self, 
        coronary_type: Literal["LCA", "RCA"] = "LCA",
        gray_reverse: bool = True
    ) -> tuple[torch.Tensor, dict]:
        """
        Run whole process of rotate DRR
        Returns:
            np.ndarray: shape = (t, c, h, w), t = rotate_parameters.total_frame, c = 1
            dict: geometry information
        """
        assert coronary_type in ["LCA", "RCA"]
        total_frame = self.drr.rotate_cfg.total_frame
        w, h = self.drr.image_size
        frames = torch.zeros(total_frame, 1, w, h)
        for f in tqdm(range(total_frame), desc="Generating Rotate DSA..."):
            phase = self._get_phase_at_frame(f)
            read_res = self.reader.get_data(phase)
            coronary = read_res.lca_label if coronary_type == "LCA" else read_res.rca_label
            
            volume = self.constrast_sim.simulate(
                ori_volume=read_res.volume,
                cavity_label=read_res.cavity_label,
                coronary_label=coronary
            )
            drr = self.drr.get_projection_at_frame(
                frame=f,
                volume=volume,
                coronary=coronary,
                affine=read_res.affine
            )
            frames[f] = drr.cpu()
        
        frames = ((frames - frames.min()) / (frames.max() - frames.min()))*255
        frames = frames.to(torch.uint8)
        if gray_reverse:
            frames = torch.tensor(255) - frames
        return frames, self.get_geometry_json()
    
    def run_and_save(
        self, 
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"],
        base_name: str = "rotate_dsa",
        gray_reverse: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        frames, json_data = self.run(coronary_type, gray_reverse)
        frames_new = frames.transpose(-1, -2).flip(-2)
        save_tif(output_dir / f"{base_name}.tif", frames_new)
        save_gif(output_dir / f"{base_name}.gif", frames_new)
        save_pngs(output_dir / f"{base_name}", frames_new)
        with open(output_dir / f"{base_name}.json", "w") as f:
            json.dump(json_data, f)
        return frames, json_data
    
    def _get_phase_at_frame(self, frame: int) -> CardiacPhase:
        return CardiacPhase.from_time(
            frame / self.drr.rotate_cfg.fps, 
            self.physical_config.cardiac_cycle_time
        )
    
    def get_geometry_json(self) -> dict:
        assert self.drr.label_center_voxel is not None
        res = {}
        res["origin_image_size"] = self.reader.origin_image_size
        res["origin_image_affine"] = self.reader.origin_image_affine.tolist()
        res["c_arm_geometry"] = self.drr.c_arm_cfg.to_dict()
        res["rotate_parameters"] = self.drr.rotate_cfg.to_dict()
        res["label_center_voxel"] = self.drr.label_center_voxel
        if (additional_config := self.drr.get_additional_config()):
            res["additional_config"] = additional_config
        res["frames"] = []
        
        for f in range(self.drr.rotate_cfg.total_frame):
            R, T = self.drr.get_R_T_at_frame(f)
            R_w2c = R.squeeze().T
            T_w2c = - R_w2c @ T.squeeze()
            d = {
                "time_s": f / self.drr.rotate_cfg.fps,
                "frame": f,
                "phase": float(self._get_phase_at_frame(f)),
                "angle": self.drr.rotate_cfg.get_rotation_angle_at_frame(f),
                "R_w2c": R_w2c.squeeze().tolist(),
                "T_w2c": T_w2c.squeeze().tolist()
            }
            
            res["frames"].append(d)
        
        return res