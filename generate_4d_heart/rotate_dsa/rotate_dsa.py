from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any
import json
import math
import threading
import queue
from concurrent.futures import ProcessPoolExecutor

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
from .torch3d_label_renderer import Torch3DLabelRenderer

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
        Run whole process of rotate DRR.
        
        Two-stage pipeline:
        1. Batch-render the STATIC (non-warped, preprocessed) volume at all
           frame angles with a single DRR _setup() call.
        2. Per frame: render the DIFFERENCE volume (warped_full - static_full)
           at the same centering affine, then add to the static result.
           DRR is linear, so static + diff = full warped DRR.
           Diff is ~sparse (nonzero only in ROI) and clamped ≥ 0 to avoid
           unphysical brightness from negative line integrals.
        
        Returns:
            RotateDsaOutput: Contains all generated outputs and metadata
        """
        assert coronary_type in ["LCA", "RCA"]
        total_frame = self.drr.rotate_cfg.total_frame
        w, h = self.drr.image_size
        device = self.drr.device

        # Pre-allocate results on GPU
        frames = torch.zeros(total_frame, w, h, device=device, dtype=torch.float32)
        labels = np.zeros((total_frame, w, h), dtype=np.uint8)
        depth_maps = np.zeros((total_frame, w, h), dtype=np.float32)

        # ── Stage 1: Batch-render static volume (WITH ROI) at all angles ──
        # Static provides the full background. Per-frame diff fills in ROI changes.
        static_vol, centering_affine, _roi_mask = self.reader.get_static_data(coronary_type)
        static_vol_gpu = static_vol.to(device)
        empty_coronary = torch.zeros_like(static_vol_gpu, dtype=torch.bool)
        static_frames = self.drr.get_projections_at_frames(
            list(range(total_frame)),
            volume=static_vol_gpu,
            coronary=empty_coronary,
            affine=centering_affine,
        )
        if static_frames.dim() == 3 and static_frames.shape[1] == 1:
            static_frames = static_frames.squeeze(1)

        # ── Stage 2: Per-frame render difference volume (warped - static) ──
        # diff = warped_full - static_full  →  nonzero only inside ROI (sparse)
        # static_frames + DRR(diff) = DRR(warped_full)  (by linearity of line integrals)
        task_queue: queue.Queue = queue.Queue(maxsize=2)

        def _producer():
            try:
                for f in range(total_frame):
                    phase = self._get_phase_at_frame(f)
                    time = f / self.drr.rotate_cfg.fps
                    read_res = self.reader.get_data(phase, coronary_type, time)
                    task_queue.put((f, read_res))
            except Exception as e:
                task_queue.put(e)
            finally:
                task_queue.put(None)

        prod_thread = threading.Thread(target=_producer, daemon=True)
        prod_thread.start()

        for _ in tqdm(range(total_frame), desc="Generating Rotate DSA..."):
            item = task_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            f, read_res = item
            coronary = read_res.coronary

            # diff = warped_full - static_full  (nonzero only inside ROI, sparse)
            # Clip negative values: motion can reduce density locally, but
            # negative line integrals → exp(-negative) > 1 (unphysical bright artifact)
            diff_vol = (coronary.volume - static_vol_gpu).clamp(min=0.0)

            drr_res = self.drr.get_projection_at_frame(
                frame=f,
                volume=diff_vol,
                coronary=coronary.label,
                affine=coronary.centering_affine,
            )

            l, dm = self.label_plotter.render_silhouette_depth(
                coronary.mesh_centering,
                *self.drr.get_R_T_at_frame(f),
            )
            labels[f] = l
            depth_maps[f] = dm

            artifacts = coronary.artifacts_centering
            if artifacts is not None and len(artifacts) > 0:
                art_integral = self.label_plotter.render_line_integral(
                    artifacts,
                    *self.drr.get_R_T_at_frame(f),
                )
                drr_res += torch.from_numpy(art_integral)[None][None].to(drr_res.device)

            # static_no_roi + diff → background outside ROI + dynamic change inside ROI
            frames[f] = static_frames[f] + drr_res

        prod_thread.join()

        # Beer-Lambert: line integral → intensity
        frames = torch.exp(-frames)
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
        Run whole process of rotate DRR (labels + depth only, no X-ray projection).
        
        Uses a producer thread for data loading and the main thread for GPU rendering,
        with a bounded queue (maxsize=2) for pipelined parallelism.
        
        Returns:
            np.ndarray: coronary labels, shape is the same as DRR images, 0: background, 1: coronary
            np.ndarray: depth masks, shape is the same as DRR images, nan for background, depth value for coronary
        """
        assert coronary_type in ["LCA", "RCA"]
        total_frame = self.drr.rotate_cfg.total_frame
        w, h = self.drr.image_size

        labels = np.zeros((total_frame, w, h), dtype=np.uint8)
        depth_maps = np.zeros((total_frame, w, h), dtype=np.float32)

        task_queue: queue.Queue = queue.Queue(maxsize=2)

        # --- Producer thread: read data ---
        def _producer():
            try:
                for f in range(total_frame):
                    phase = self._get_phase_at_frame(f)
                    read_res = self.reader.get_data(phase, coronary_type)
                    task_queue.put((f, read_res))
            except Exception as e:
                task_queue.put(e)
            finally:
                task_queue.put(None)

        prod_thread = threading.Thread(target=_producer, daemon=True)
        prod_thread.start()

        # --- Consumer (main thread): render labels + depth on GPU ---
        for _ in tqdm(range(total_frame), desc="Generating Rotate DSA without DRR..."):
            item = task_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            f, read_res = item
            coronary = read_res.coronary

            l, dm = self.label_plotter.render_silhouette_depth(
                coronary.mesh_centering,
                *self.drr.get_R_T_at_frame(f),
            )
            labels[f] = l
            depth_maps[f] = dm

        prod_thread.join()
        return labels, depth_maps
    
    
    def run_and_save(
        self, 
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"],
        gif_fps: int = 30   # gif may not support too high fps (like fps=60 may cause gif slow)
    ) -> RotateDsaOutput:
        output = self.run(coronary_type)
        
        phase_0_data = self.reader.get_phase_0_data(coronary_type)

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(save_tif, output_dir / "rotate_dsa_raw.tif", output.raw_frames),
                executor.submit(save_gif, output_dir / "rotate_dsa.gif", output.vis_frames, gif_fps, cmap='gray', vmin=0, vmax=255),
                executor.submit(save_gif, output_dir / "rotate_dsa_raw.gif", output.raw_frames, gif_fps, cmap='gray'),
                executor.submit(save_gif, output_dir / "label.gif", output.labels * 255, gif_fps, cmap='gray'),
                executor.submit(save_deepthmap_gif, output_dir / "depth_map.gif", output.depth_maps, gif_fps),
                executor.submit(save_pngs, output_dir / "rotate_dsa", output.vis_frames),
                executor.submit(save_pngs, output_dir / "label", output.labels * 255),
                executor.submit(np.savez_compressed, output_dir / "label.npz", output.labels),
                executor.submit(np.savez_compressed, output_dir / "depth_map.npz", output.depth_maps),
            ]
            # phase_0_data.save() 含 torch 张量，在当前进程运行，与子进程池并行
            meta = phase_0_data.save(
                output_dir,
                save_nii_type=["coronary"], save_mesh=True, save_central_line=True,
                coordinate_system="coronary_centering"
            )
            for f in futures:
                f.result()
        
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
        phase_0_data = self.reader.get_phase_0_data(coronary_type)

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(save_gif, output_dir / "label.gif", labels * 255, gif_fps, cmap='gray'),
                executor.submit(save_deepthmap_gif, output_dir / "depth_map.gif", depth_maps, gif_fps),
                executor.submit(save_pngs, output_dir / "label", labels * 255),
                executor.submit(np.savez_compressed, output_dir / "label.npz", labels),
                executor.submit(np.savez_compressed, output_dir / "depth_map.npz", depth_maps),
            ]
            # phase_0_data.save() 含 torch 张量，在当前进程运行
            meta = phase_0_data.save(
                output_dir,
                save_nii_type=["coronary"], save_mesh=True, save_central_line=True,
                coordinate_system="coronary_centering"
            )
            for f in futures:
                f.result()
        
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