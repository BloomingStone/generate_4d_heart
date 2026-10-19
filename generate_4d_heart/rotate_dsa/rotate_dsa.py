from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any
import json
import threading
import queue
import warnings
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import numpy as np
import torch


from .data_reader import DataReader, DataReaderResult
from .rotate_drr import RotateDRR
from ..med_image import MedImage
from ..types import Sec
from .cardiac_phase import CardiacPhase
from ..saver import save_tif, save_gif, save_pngs, save_deepthmap_gif, save_video, save_deepthmap_video
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
    artifacts_proj: np.ndarray|None
    meta: dict[str, Any]


@dataclass
class RotateDSA:
    reader: DataReader
    drr: RotateDRR
    physical_config: PhysicalConfig = field(default_factory=PhysicalConfig)
    multi_threads: bool = True
    """When False, disable producer thread and run single-threaded for easier debugging."""
    
    label_plotter: Torch3DLabelRenderer = field(init=False)
    
    def __post_init__(self):
        self.label_plotter = Torch3DLabelRenderer(self.drr.c_arm_cfg)

    def run(
        self, 
        coronary_type: Literal["LCA", "RCA"] = "LCA",
    ) -> RotateDsaOutput:
        """
        Run whole process of rotate DRR.
        
        Multi-layer pipeline:
        1. Batch-render static layers (background chest CT, etc.) at all frame angles.
           Each layer has its own volume+affine; regions covered by later layers
           are zeroed out to avoid double-counting.
        2. Per frame: render the warped heart volume (with contrast + motion).
        3. All line integrals sum linearly; Beer-Lambert applied at the end.
        
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
        artifacts_proj = np.zeros((total_frame, w, h), dtype=np.float32)
        has_artifacts = False

        # ── Stage 1: Batch-render all static layers ──
        for vol_img in self.reader.get_static_data(coronary_type):
            layer_frames = self.drr.get_projections_at_frames(
                list(range(total_frame)),
                volume=vol_img.to(device=device),
                coronary=MedImage(
                    data=torch.zeros_like(vol_img.data, dtype=torch.bool),
                    affine=vol_img.affine,
                    coord=vol_img.coord,
                ),
            )
            if layer_frames.dim() == 4 and layer_frames.shape[1] == 1:
                layer_frames = layer_frames.squeeze(1)
            frames += layer_frames

        # ── Stage 2: Per-frame dynamic rendering ──

        def _render_one_frame(f: int, read_res: DataReaderResult):
            coronary = read_res.coronary

            if f == 0:
                print(f"[dynamic layer] shape={coronary.volume.data.shape}, centering_affine[:3,3]={coronary.centering_affine[:3,3]}, "
                      f"T_orig_to_centering={coronary.T_original_to_centering}")

            drr_res = self.drr.get_projection_at_frame(
                frame=f,
                volume=coronary.volume_centering,
                coronary=coronary.label_centering,
            )
            # drr_res shape: (1, 1, H, W) → squeeze to (H, W)
            drr_res = drr_res.squeeze()

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
                drr_res += torch.from_numpy(art_integral).to(drr_res.device)
                artifacts_proj[f] += art_integral
                nonlocal has_artifacts
                has_artifacts = True

            frames[f] += drr_res


        if self.multi_threads:
            task_queue: queue.Queue = queue.Queue(maxsize=2)

            def _producer():
                try:
                    for f in range(total_frame):
                        phase = self._get_phase_at_frame(f)
                        time = f / self.drr.rotate_cfg.fps
                        read_res = self.reader.get_dynamic_data(phase, coronary_type, time)
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
                _render_one_frame(f, read_res)  # projector at main thread, data loading at producer thread

            prod_thread.join()
        else:
            for f in tqdm(range(total_frame), desc="Generating Rotate DSA (single-thread)..."):
                phase = self._get_phase_at_frame(f)
                time = f / self.drr.rotate_cfg.fps
                read_res = self.reader.get_dynamic_data(phase, coronary_type, time)
                _render_one_frame(f, read_res)

        # Beer-Lambert: line integral → intensity
        frames = torch.exp(-frames)
        vis_frames, post_process_meta = postprocess_drr(frames)

        return RotateDsaOutput(
            raw_frames=frames.cpu().numpy().astype(np.float32),
            vis_frames=vis_frames.cpu().numpy().astype(np.uint8),
            labels=labels,
            depth_maps=depth_maps,
            artifacts_proj=artifacts_proj if has_artifacts else None,
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

        def _render_one_frame_no_drr(f: int, read_res: DataReaderResult):
            coronary = read_res.coronary
            l, dm = self.label_plotter.render_silhouette_depth(
                coronary.mesh_centering,
                *self.drr.get_R_T_at_frame(f),
            )
            labels[f] = l
            depth_maps[f] = dm

        if self.multi_threads:
            task_queue: queue.Queue = queue.Queue(maxsize=2)

            def _producer():
                try:
                    for f in range(total_frame):
                        phase = self._get_phase_at_frame(f)
                        read_res = self.reader.get_dynamic_data(phase, coronary_type)
                        task_queue.put((f, read_res))
                except Exception as e:
                    task_queue.put(e)
                finally:
                    task_queue.put(None)

            prod_thread = threading.Thread(target=_producer, daemon=True)
            prod_thread.start()

            for _ in tqdm(range(total_frame), desc="Generating Rotate DSA without DRR..."):
                item = task_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                f, read_res = item
                _render_one_frame_no_drr(f, read_res)

            prod_thread.join()
        else:
            for f in tqdm(range(total_frame), desc="Generating Rotate DSA without DRR (single-thread)..."):
                phase = self._get_phase_at_frame(f)
                read_res = self.reader.get_dynamic_data(phase, coronary_type)
                _render_one_frame_no_drr(f, read_res)

        return labels, depth_maps
    
    
    def run_and_save(
        self, 
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"],
        gif_fps: int = 30,   # gif may not support too high fps (like fps=60 may cause gif slow)
        phase_0_save_types: tuple[DataReaderResult.SupportedSaveType, ...] = ("coronary", "cor-mesh", "cor-artifacts", "central-line"),
        phase_0_save_coordinate: DataReaderResult.SupportedSaveCoordinate = "centering",
        prefix: str = "rotate_dsa",
        extra_json: dict[str, Any] | None = None,
    ) -> RotateDsaOutput:
        """跑完整流程并落盘。

        `prefix` 控制主产物命名（`{prefix}_raw.tif` / `{prefix}.gif` / `{prefix}.json` 等），
        默认 `"rotate_dsa"`（行为不变）；固定视角模式用 `"fix_angle_dsa"`。
        `extra_json` 会合并进最后写出的 `{prefix}.json`（如 `{"fixed_angle": {...}}`）。
        """
        output = self.run(coronary_type)
        
        video_fps = self.drr.rotate_cfg.fps
        total_frame = self.drr.rotate_cfg.total_frame

        # 后端可选的辐射剂量估计（目前仅 xcist 后端实现；
        # 未启用 dose_cfg 时 estimate_run_dose() 返回 None，无任何副作用）。
        # 注意：剂量失败**不应**影响 DSA 图像落盘，因此只告警不抛。
        dose_result = None
        dose_estimator = getattr(self.drr, "estimate_run_dose", None)
        if callable(dose_estimator):
            try:
                dose_result = dose_estimator()
            except Exception as exc:            # noqa: BLE001
                warnings.warn(f"剂量估计失败，已跳过剂量输出：{type(exc).__name__}: {exc}")

        output_dir.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(save_tif, output_dir / f"{prefix}_raw.tif", output.raw_frames),
                executor.submit(save_pngs, output_dir / f"{prefix}", output.vis_frames),
                executor.submit(save_pngs, output_dir / "label", output.labels * 255),
                executor.submit(np.savez_compressed, output_dir / "label.npz", output.labels),
                executor.submit(np.savez_compressed, output_dir / "depth_map.npz", output.depth_maps),
            ]
            if dose_result is not None:
                dose_arr = np.ascontiguousarray(dose_result["dose"], dtype=np.float32)
                futures.append(executor.submit(np.save, output_dir / "dose.npy", dose_arr))
                # 同时写 gecatsim 官方 .mGy（即 raw float32）
                futures.append(executor.submit(dose_arr.tofile, output_dir / "dose.mGy"))


            if total_frame > 60:
                futures.extend([
                    executor.submit(save_video, output_dir / f"{prefix}.mp4", output.vis_frames, video_fps, vmin=0, vmax=255),
                    executor.submit(save_video, output_dir / f"{prefix}_raw.mp4", output.raw_frames, video_fps),
                    executor.submit(save_video, output_dir / "label.mp4", output.labels * 255, video_fps),
                    executor.submit(save_deepthmap_video, output_dir / "depth_map.mp4", output.depth_maps, video_fps),
                ])
                if output.artifacts_proj is not None:
                    futures.append(
                        executor.submit(save_video, output_dir / "artifacts_proj.mp4", output.artifacts_proj, video_fps)
                    )
            else:
                futures.extend([
                    executor.submit(save_gif, output_dir / f"{prefix}.gif", output.vis_frames, gif_fps, cmap='gray', vmin=0, vmax=255),
                    executor.submit(save_gif, output_dir / f"{prefix}_raw.gif", output.raw_frames, gif_fps, cmap='gray'),
                    executor.submit(save_gif, output_dir / "label.gif", output.labels * 255, gif_fps, cmap='gray'),
                    executor.submit(save_deepthmap_gif, output_dir / "depth_map.gif", output.depth_maps, gif_fps, cmap='gray'),
                ])
                if output.artifacts_proj is not None:
                    futures.append(
                        executor.submit(save_gif, output_dir / "artifacts_proj.gif", output.artifacts_proj, gif_fps, cmap='gray')
                    )
            
            # phase_0_data.save() 含 torch 张量，在当前进程运行，与子进程池并行
            if phase_0_save_types:
                phase_0_data = self.reader.get_phase_0_data(coronary_type)
                phase_0_meta = phase_0_data.save(
                    output_dir,
                    save_types=phase_0_save_types,
                    coordinate_system=phase_0_save_coordinate
                )
            else:
                phase_0_meta = {}
            
            for f in futures:
                f.result()
        
        json_data = self.get_geometry_json(coronary_type)
        json_data["postprocess_meta"] = output.meta
        json_data["phase_0_save_meta"] = phase_0_meta
        json_data["reader_meta"] = self.reader.meta
        if dose_result is not None:
            json_data["dose"] = dose_result["meta"]
        if extra_json:
            json_data.update(extra_json)
        with open(output_dir / f"{prefix}.json", "w") as f:
            json.dump(json_data, f, indent=2)
        
        return output
    
    
    def run_and_save_no_drr(
        self, 
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"],
        gif_fps: int = 30,   # gif may not support too high fps (like fps=60 may cause gif slow)
        phase_0_save_types: tuple[DataReaderResult.SupportedSaveType, ...] = ("coronary", "cor-mesh", "cor-artifacts", "central-line"),
        phase_0_save_coordinate: DataReaderResult.SupportedSaveCoordinate = "centering"
    ) -> tuple[np.ndarray, np.ndarray]:
        labels, depth_maps = self.run_no_drr(coronary_type)
        phase_0_data = self.reader.get_phase_0_data(coronary_type)
        video_fps = self.drr.rotate_cfg.fps
        total_frame = self.drr.rotate_cfg.total_frame

        output_dir.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(save_pngs, output_dir / "label", labels * 255),
                executor.submit(np.savez_compressed, output_dir / "label.npz", labels),
                executor.submit(np.savez_compressed, output_dir / "depth_map.npz", depth_maps),
            ]

            if total_frame > 60:
                futures.extend([
                    executor.submit(save_video, output_dir / "label.mp4", labels * 255, video_fps),
                    executor.submit(save_deepthmap_video, output_dir / "depth_map.mp4", depth_maps, video_fps),
                ])
            else:
                futures.extend([
                    executor.submit(save_gif, output_dir / "label.gif", labels * 255, gif_fps, cmap='gray'),
                    executor.submit(save_deepthmap_gif, output_dir / "depth_map.gif", depth_maps, gif_fps),
                ])
            
            # phase_0_data.save() 含 torch 张量，在当前进程运行
            if phase_0_save_types:
                phase_0_meta = phase_0_data.save(
                    output_dir,
                    save_types=phase_0_save_types,
                    coordinate_system=phase_0_save_coordinate
                )
            else:
                phase_0_meta = {}
            
            for f in futures:
                f.result()
        
        json_data = self.get_geometry_json(coronary_type)
        json_data["phase_0_save_meta"] = phase_0_meta
        json_data["reader_meta"] = self.reader.meta
        with open(output_dir / "rotate_dsa.json", "w") as f:
            json.dump(json_data, f, indent=2)
        
        return labels, depth_maps
    
    def save_no_run(
        self, 
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"],
        phase_0_save_types: tuple[DataReaderResult.SupportedSaveType, ...] = ("coronary", "cor-mesh", "cor-artifacts", "central-line"),
        phase_0_save_coordinate: DataReaderResult.SupportedSaveCoordinate = "centering",
    ) -> None:
        if phase_0_save_types:
            phase_0_data = self.reader.get_phase_0_data(coronary_type)
            phase_0_meta = phase_0_data.save(
                output_dir, 
                save_types=phase_0_save_types,
                coordinate_system=phase_0_save_coordinate
            )
        else:
            phase_0_meta = {}
        
        json_data = self.get_geometry_json(coronary_type)
        json_data["phase_0_save_meta"] = phase_0_meta
        json_data["reader_meta"] = self.reader.meta
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