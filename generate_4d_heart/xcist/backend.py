"""XCIST 正式后端编排器（骨架：静态案例可运行，动态/标签叠加后续）。

`XcistBackend` 是独立于 `RotateDSA/TorchDRR` 的第二条投影管线：
整卷逐帧重投影（多能谱 + 平板），复用 `postprocess_drr`/`saver` 输出
RotateDSA 同构文件。设计见同目录 `backend_design.md`。

静态路径用法（单 ROI CTA，无运动/造影/伪影）：
    backend = XcistBackend(geometry=geom)
    backend.run_and_save_static(
        out_dir, hu=hu, affine=aff, iso=iso,
        alphas_deg=[-20, 0, 20],
    )

TODO(S3/S4)：
  - fusion.py：reader 静态层+动态层融合成"完整世界 HU 卷"（iso 原点公共网格）
  - 材料：按冠状动脉标签加 iodine、伪影 mesh 体素化（bone/金属材料）
  - 每帧 R/T 写入 json；Torch3DLabelRenderer 叠加 label/depth（复用同一 R/T）
  - 多帧批量（startAngle 步进）替代逐帧单 view 以省开销
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch

from .geometry import XcistGeometry
from .phantom import VoxelPhantom, build_from_ct
from .sim import XcistSim, XcistPhysics
from ..rotate_dsa.postprocess import postprocess_drr
from ..saver import save_tif, save_gif, save_pngs


@dataclass
class FrameInfo:
    alpha_degree: float
    beta_degree: float = 0.0
    phase: float = 0.0
    time_s: float = 0.0


class XcistBackend:
    """整卷逐帧 xcist 投影后端。"""

    def __init__(
        self,
        geometry: XcistGeometry,
        physics: XcistPhysics | None = None,
        phantom_out_dir: Optional[str] = None,
    ):
        self.geometry = geometry
        self.sim = XcistSim(geometry=geometry, physics=physics,
                            phantom_out_dir=phantom_out_dir)
        self._phantom_json: Optional[str] = None

    # ------------------------------------------------------------------
    def set_static_phantom(
        self,
        hu: np.ndarray,
        affine: np.ndarray,
        iso: Optional[np.ndarray] = None,
        *,
        half_mm=(60.0, 60.0, 55.0),
        voxel_mm=(0.6, 0.6, 0.625),
        name="gen4d_static",
    ) -> VoxelPhantom:
        """从静态 CTA 构建并注册体模（iso 缺省=体积中心）。"""
        phantom = build_from_ct(hu, affine, iso, half_mm=half_mm,
                                voxel_mm=voxel_mm, name=name)
        self._phantom_json = phantom.write(self.sim.phantom_out_dir)
        return phantom

    def render_frames(
        self,
        alphas_deg: list[float],
        *,
        as_transmission: bool = True,
    ) -> np.ndarray:
        """逐帧投影，返回 (T,H,W) 透射或探测器计数（float32）。"""
        assert self._phantom_json is not None, "call set_static_phantom first"
        frames = []
        for a in alphas_deg:
            if as_transmission:
                frames.append(self.sim.transmission(a, self._phantom_json))
            else:
                frames.append(self.sim.project(a, self._phantom_json))
        return np.stack(frames, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    def run_and_save_static(
        self,
        out_dir,
        hu: np.ndarray,
        affine: np.ndarray,
        iso: Optional[np.ndarray] = None,
        *,
        alphas_deg: list[float] = (0.0,),
        gif_fps: int = 10,
        gamma: float = 0.1,
        save_png_frames: bool = False,
        physics_meta: Optional[dict] = None,
    ) -> Path:
        """静态案例完整输出：raw.tif / gif / pngs / json。"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        phantom = self.set_static_phantom(hu, affine, iso)
        alphas = list(alphas_deg)

        # 1) 投影 → 透射 frames (T,H,W)
        frames_t = self.render_frames(alphas, as_transmission=True)
        # 2) raw tif（float32 透射）
        save_tif(out_dir / "rotate_dsa_raw.tif", frames_t)
        # 3) 后处理可视化（复用 postprocess_drr，输出 uint8）
        vis, ppm = postprocess_drr(torch.from_numpy(frames_t).float(),
                                   gamma=gamma)
        save_gif(out_dir / "rotate_dsa.gif", vis, fps=gif_fps)
        if save_png_frames:
            save_pngs(out_dir / "rotate_dsa", vis)

        # 4) 几何/物理/逐帧 json
        g = self.geometry
        c_arm = {
            "sdd": g.sdd, "sod": g.sid,
            "height": g.n_row, "width": g.n_col,
            "delx": g.col_size, "dely": g.row_size,
            "x0": 0.0, "y0": 0.0,
            "orientation_type": g.orientation,
            "renderer": "xcist(gecatsim) flat-panel poly",
        }
        p = self.sim.physics
        phys = {
            "spectrum_filename": p.spectrum_filename,
            "flat_filter_mm": p.flat_filter_mm,
            "mA": p.mA,
            "exposure_s": p.exposure_s,
            "quantum_noise": p.enable_quantum_noise,
            "electronic_noise": p.enable_electronic_noise,
        }
        meta = {
            "coronary_type": "LCA(static-demo)",
            "c_arm_geometry": c_arm,
            "rotate_parameters": {
                "fps": gif_fps,
                "total_frame": len(alphas),
                "coordinate_system": "RAS",
                "parameterization": "euler_angles",
                "convention": "ZXY",
                "startAngle_from_alpha": True,
            },
            "physics": phys,
            "phantom": {
                "materials": list(phantom.arrays.keys()),
                "shape": phantom.shape,
                "voxel_mm": phantom.voxel_mm,
            },
            "postprocess_meta": ppm,
            "frames": [asdict(FrameInfo(alpha_degree=a)) for a in alphas],
        }
        if physics_meta:
            meta.update(physics_meta)
        with open(out_dir / "rotate_dsa.json", "w") as f:
            json.dump(meta, f, indent=2, allow_nan=True)
        print(f"[xcist-backend] saved -> {out_dir}")
        return out_dir
