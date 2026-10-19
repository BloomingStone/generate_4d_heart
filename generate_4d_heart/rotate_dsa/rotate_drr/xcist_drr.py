"""XCIST (gecatsim) 作为 `RotateDRR` 的并行实现。

两种模式：

- ``mode="mono"``（单能，**真正的 drop-in**）：
  xcist 配置为 `physics.monochromatic=E`，此时透射 ``T=exp(-∫μds)``，
  本类返回 ``E=-ln T``（线积分），与 `TorchDRR` 数值同构。因此
  `RotateDSA` 的"多层线积分相加 + 末尾 exp(-frames)"架构**原样适用**，
  静态层的批量优化也保留（`get_projections_at_frames` 一次多 view 投影）。
  输入 μ → 单材料水当量密度 ρ=μ/μ_water(E)，精确复现原 μ。

- ``mode="poly"``（多能 + 平板，**高保真但非 drop-in**）：
  xcist 用真实能谱（如 90kVp+0.1mmCu）。由于能谱+探测非线性，
  多层线积分不可相加，故采用"延迟融合"：
    * `get_projections_at_frames` 只缓存静态层体积，返回全 0；
    * `get_projection_at_frame` 把静态层 + 当前帧动态层融合为**单一材料体**，
      整卷投影，返回 ``E_eff=-ln T``（使 `RotateDSA` 末尾 exp 仍还原为 T）。
  输入 μ → {water, bone, iodine} 材料密度（用 `coronary` 标签分碘）。

注意：artifact 的 mesh 线积分仍由 `RotateDSA` 在 log 域叠加（近似）；
若要严格一致应把伪影体素化为金属材料（后续）。

辐射剂量（**可选**，默认关闭 → 零额外耗时/零额外文件）：

    传入 `dose_cfg=DoseConfig(...)` 启用后，`RotateDSA.run_and_save` 会同步
    多输出 `dose.npy` / `dose.mGy`，并把统计写进 `rotate_dsa.json["dose"]`：

        drr = XcistDRR(c_arm_cfg=geom, rotate_cfg=rcfg, mode="mono",
                       dose_cfg=DoseConfig(n_voxel=64, fov_mm=256, slice_thickness_mm=4),
                       dose_subsample=12, dose_slices=32)
        drr.set_dose_reference_hu(hu_xyz)   # 推荐：参考 CT，HU，(64,64,S)，iso 原点
        RotateDSA(reader=reader, drr=drr).run_and_save(out_dir, "LCA")

    * 剂量沉积介质：优先用 `set_dose_reference_hu` 提供的参考 CT
      (`hu_source="reference_ct"`)；未提供时回退到 DRR 收到的 μ 体积反推 HU
      (`hu_source="mu_approx"`，只含心脏 ROI，偏低估)。
    * 与投影模式解耦：即使 `mode="mono"`，剂量也用**真实能谱**
      （`dose_energy_bins` 个能量 bin + `flat_filter` 滤过），而非单能。
    * 估计口径：按协议均匀取 `dose_subsample` 个视角逐 view 沉积，再按
      总帧数/采样数线性缩放（`dose_subsample=1` = 全协议逐帧，最准最慢）；
      每 view 束流 = 一帧曝光 `exposure_s`，总 mAs = mA×exposure_s×总帧数。
    * 是"被照体积内累积吸收剂量"，非 DAP/CTDI；材料仅 water/bone。

    同一份 `meta` 里还会带一个 **CTDI 口径**（`meta["ctdi"]`，由
    `estimate_ctdi()`/`dose.estimate_ctdi_like()` 算）：均匀水当量/PMMA 圆柱体模、
    360° 单圈、中央 100mm 平均，给出 `mean100` / `ctdi100` / `ctdi_w` / `per_100mAs`。
    参数用 `ctdi_cfg=CtdiConfig(...)`，`compute_ctdi=False` 可关掉；
    它与患者剂量（whole/iso/roi）**并列**写在 json 里，口径不同不可相减。
"""
from __future__ import annotations

import numpy as np
import torch
from dataclasses import replace
from diffdrr.pose import convert, RigidTransform

from .rotated_drr import CArmGeometry, RotatedParameters
from .torch_drr import get_reorientation
from ...xcist.geometry import XcistGeometry
from ...xcist.materials import (
    mu_of_material_mm, mu_to_materials, mu_to_water_density,
    resample_scalar_to_canonical, sum_materials, union_canonical_grid,
)
from ...xcist.phantom import VoxelPhantom
from ...xcist.sim import XcistPhysics, XcistSim
from ...xcist.dose import (
    CtdiConfig, DoseConfig, estimate_air_kerma, estimate_ctdi_like, estimate_dose,
    max_dose_ref,
)

_MODES = ("mono", "poly")


def _as_numpy_3d(img) -> np.ndarray:
    """MedImage.data → (N0,N1,N2) numpy（去掉前导 batch/channel 维）。"""
    t = img.data if hasattr(img, "data") else img
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    arr = np.asarray(t)
    return np.ascontiguousarray(arr.reshape(arr.shape[-3:]))


class XcistDRR:
    """`RotateDRR` 的 xcist 实现（mono drop-in / poly 融合）。"""

    def __init__(
        self,
        c_arm_cfg: CArmGeometry = CArmGeometry(),
        rotate_cfg: RotatedParameters = RotatedParameters(),
        physics: XcistPhysics | None = None,
        *,
        mode: str = "mono",
        monochromatic_kev: float = 70.0,
        reference_kev: float = 70.0,
        grid_voxel_mm: tuple[float, float, float] | None = None,
        name: str = "xcist_drr",
        dose_cfg: DoseConfig | None = None,
        dose_subsample: int = 12,
        dose_slices: int = 32,
        dose_energy_bins: int = 20,
        ctdi_cfg: CtdiConfig | None = None,
        compute_ctdi: bool = True,
    ):
        assert mode in _MODES, f"mode must be one of {_MODES}, got {mode!r}"
        self.c_arm_cfg = c_arm_cfg
        self.rotate_cfg = rotate_cfg
        self.mode = mode
        self.monochromatic_kev = float(monochromatic_kev)
        self.reference_kev = float(reference_kev)
        self.grid_voxel_mm = grid_voxel_mm
        self.name = name

        self.geometry = XcistGeometry.from_c_arm(c_arm_cfg)
        phys = physics or XcistPhysics()
        if mode == "mono":
            phys.monochromatic_kev = float(monochromatic_kev)
        self.physics = phys
        self.sim = XcistSim(geometry=self.geometry, physics=phys)

        # 相机位姿参数（与 TorchDRR 完全一致，保证与 Torch3D 标签方案对齐）
        self.reorient = get_reorientation(c_arm_cfg.orientation_type)
        sod = c_arm_cfg.sod
        if c_arm_cfg.orientation_type == "AP":
            self.translations = torch.tensor([[0.0, sod, 0.0]])
        elif c_arm_cfg.orientation_type == "PA":
            self.translations = torch.tensor([[0.0, -sod, 0.0]])
        else:
            raise ValueError(f"orientation_type must be AP/PA, got {c_arm_cfg.orientation_type}")

        # poly 状态
        self._static_layers: list[tuple[np.ndarray, np.ndarray]] = []
        self._static_mats: dict[str, np.ndarray] | None = None
        self._grid: tuple[np.ndarray, tuple, tuple] | None = None
        self._dynamic_seen = False   # 同一实例多次 run 时用于自动重置缓存

        # 剂量（可选，dose_cfg=None 时完全关闭、零额外耗时）
        self.dose_cfg = dose_cfg
        self.dose_subsample = int(dose_subsample)
        self.dose_slices = int(dose_slices)
        self.dose_energy_bins = int(dose_energy_bins)
        self.ctdi_cfg = ctdi_cfg
        self.compute_ctdi = bool(compute_ctdi)
        self._dose_sim: XcistSim | None = None      # 用得是**非单能**能谱（剂量需真实谱）
        self._dose_hu_ref: np.ndarray | None = None  # 调用方提供的参考 CT(HU)
        self._dose_hu_fallback: np.ndarray | None = None
        self._kerma_sim: XcistSim | None = None      # Ka,r/DAP 专用（只需谱）

    # ------------------------------------------------------------------ 协议
    @property
    def image_size(self) -> tuple[int, int]:
        return (self.c_arm_cfg.height, self.c_arm_cfg.width)

    @property
    def device(self) -> torch.device:
        # gecatsim 为 CPU 计算；保持 CPU 使 RotateDSA 的 frames/体积都在 CPU
        return torch.device("cpu")

    def get_additional_config(self) -> dict:
        p = self.physics
        return {
            "renderer": "xcist",
            "mode": self.mode,
            "monochromatic_kev": (self.monochromatic_kev if self.mode == "mono" else None),
            "reference_kev": self.reference_kev,
            "spectrum_filename": p.spectrum_filename,
            "flat_filter_mm": list(p.flat_filter_mm),
            "mA": p.mA,
            "exposure_s": p.exposure_s,
            "fps": p.fps,
            "quantum_noise": p.enable_quantum_noise,
            "c_arm": {"sdd": self.geometry.sdd, "sid": self.geometry.sid,
                      "n_col": self.geometry.n_col, "n_row": self.geometry.n_row,
                      "col_size": self.geometry.col_size, "row_size": self.geometry.row_size,
                      "orientation": self.geometry.orientation},
            "dose": self.dose_config_dict(),
        }

    # ------------------------------------------------------------------ 剂量
    def dose_config_dict(self) -> dict | None:
        """写入 json 的剂量配置摘要（未启用时返回 None）。"""
        if self.dose_cfg is None:
            return None
        dc = self.dose_cfg
        return {
            "enabled": True,
            "n_voxel": dc.n_voxel,
            "fov_mm": dc.fov_mm,
            "slice_thickness_mm": dc.slice_thickness_mm,
            "slices": self.dose_slices,
            "subsample_views": self.dose_subsample,
            "energy_bins": self.dose_energy_bins,
            "do_convol": dc.do_convol,
            "ctdi": (None if not self.compute_ctdi else {
                "phantom_diameter_mm": (self.ctdi_cfg or CtdiConfig()).phantom_diameter_mm,
                "integration_mm": (self.ctdi_cfg or CtdiConfig()).integration_mm,
                "periph_radius_mm": (self.ctdi_cfg or CtdiConfig()).periph_radius_mm,
            }),
            "kerma": (None if not self.compute_ctdi else {"enabled": True,
                                                       "ref_dist_mm": "sid-15"}),
        }

    def set_dose_reference_hu(self, hu_xyz: np.ndarray) -> None:
        """提供剂量沉积用的参考 CT（HU）。

        Args:
            hu_xyz: `(N, N, S)` HU 体，iso 原点、X 右 / Y 前(源侧) / Z 头向，
                且与 `dose_cfg` 的网格（n_voxel/fov_mm/slice_thickness_mm）一致。
                推荐直接用 CT，而不是 DRR 的 μ 动态体积（后者只含心脏 ROI）。
        """
        self._dose_hu_ref = np.asarray(hu_xyz, dtype=np.float32)

    def _mu_grid_to_dose_hu(self, grid_zyx: np.ndarray, voxel_mm, center,
                            kev: float) -> np.ndarray:
        """把规范化 μ 网格 (Nz,Ny,Nx) 重采样到剂量网格并反推 HU（回退路径）。

        剂量网格：iso 为原点、`(n_voxel, n_voxel, dose_slices)`、各向同性面内体素。
        `kev` 必须与 μ 网格对应的能量一致（mono=单能；poly=reference_kev）。
        """
        from ...xcist.materials import resample_scalar_to_canonical

        dc = self.dose_cfg
        assert dc is not None
        n, ns = int(dc.n_voxel), int(self.dose_slices)
        vxy = dc.fov_mm / n
        vz = float(dc.slice_thickness_mm)
        half = (dc.fov_mm / 2.0, dc.fov_mm / 2.0, ns * vz / 2.0)
        # 合成仿射：规范化数组轴序为 (z, y, x) → 列为 (Z, Y, X)
        dx, dy, dz = voxel_mm
        shape = np.asarray(grid_zyx.shape, dtype=float)
        aff = np.eye(4)
        aff[:3, :3] = np.diag([dz, dy, dx])
        aff[:3, 3] = np.asarray(center, float) - (shape - 1) / 2.0 * np.array([dz, dy, dx])
        mu = resample_scalar_to_canonical(grid_zyx, aff, (0.0, 0.0, 0.0),
                                          half, (vxy, vxy, vz), order=1)
        mu_xyz = np.ascontiguousarray(np.transpose(mu, (2, 1, 0)))   # (N,N,S)=(x,y,z)
        mu_w = float(np.asarray(__import__("gecatsim").GetMu(
            "water", [float(kev)])).ravel()[0]) / 10.0
        hu = (mu_xyz / max(mu_w, 1e-9) - 1.0) * 1000.0
        return np.clip(hu, -1000.0, 3000.0).astype(np.float32)

    def _maybe_attach_mu_fallback(self, grid_zyx, voxel_mm, center, kev) -> None:
        """未提供参考 CT 时，用 DRR 的 μ 体积反推 HU 作为剂量输入（近似）。"""
        if (self.dose_cfg is None or self._dose_hu_ref is not None
                or self._dose_hu_fallback is not None):
            return
        self._dose_hu_fallback = self._mu_grid_to_dose_hu(grid_zyx, voxel_mm, center, kev)

    def _get_dose_sim(self) -> XcistSim:
        """剂量用的仿真器：**非单能**能谱（真实谱 + 滤过），并改用真实协议能谱参数。"""
        if self._dose_sim is None:
            phys = replace(self.physics, monochromatic_kev=None,
                           energy_count=int(self.dose_energy_bins))
            self._dose_sim = XcistSim(geometry=self.geometry, physics=phys,
                                      center_offset_mm=(0.0, 0.0, 0.0))
        return self._dose_sim

    def estimate_ctdi(self, ctdi_cfg: CtdiConfig | None = None,
                      return_dose: bool = False, verbose: bool = False) -> dict:
        """按 **CTDI 口径**估一个 360° 等效旋转的剂量指数（水当量圆柱体模）。

        * 角度密度沿用本次协议：每度视角数 = 总帧数 / |扫角|，只是把扫角补成 360°；
        * 平板可降采样（`CtdiConfig.detector_downsample`）：只粗化射线采样，
          面板尺寸与总通量不变（否则在 474×648 平板上 72 视角会很慢）；
        * 用同一 physics（kVp/滤过/mA/曝光）但**非单能**谱。
        """
        c = replace(ctdi_cfg or self.ctdi_cfg or CtdiConfig())
        alphas = np.array([self._alpha_deg(f) for f in range(self.rotate_cfg.total_frame)])
        span = abs(float(alphas[-1] - alphas[0])) if alphas.size > 1 else 360.0
        span = span if span > 1e-6 else 360.0
        c.view_density_per_360 = alphas.size / span
        n_views_360 = c.n_views()
        mAs_rot = self.physics.mA * self.physics.exposure_s * n_views_360

        d = max(1, int(c.detector_downsample))
        geom = self.geometry if d == 1 else replace(
            self.geometry,
            n_col=max(8, self.geometry.n_col // d),
            n_row=max(8, self.geometry.n_row // d),
            col_size=self.geometry.col_size * d,
            row_size=self.geometry.row_size * d,
        )
        phys = replace(self.physics, monochromatic_kev=None,
                       energy_count=int(self.dose_energy_bins))
        sim = XcistSim(geometry=geom, physics=phys, center_offset_mm=(0.0, 0.0, 0.0))
        beam_w = (self.geometry.n_row * self.geometry.row_size
                  * self.geometry.sid / self.geometry.sdd)
        out = estimate_ctdi_like(sim._ct.cfg, c, mAs_per_rotation=mAs_rot,
                                 beam_width_mm=beam_w, return_dose=return_dose,
                                 verbose=verbose)
        out["detector"] = {"n_col": geom.n_col, "n_row": geom.n_row,
                           "col_size": geom.col_size, "row_size": geom.row_size,
                           "downsample": d, "panel_mm": [geom.n_col * geom.col_size,
                                                         geom.n_row * geom.row_size]}
        return out

    def estimate_air_kerma(self, n_frames: int | None = None,
                           cell: str = "center") -> dict:
        """**参考点空气比释动能 Ka,r** 与 **DAP**（自由空气口径，与设备剂量报告同量纲）。

        用**非单能**真实能谱（不受 mono 模式影响），从 CatSim 算好的
        `cfg.spec.netIvec`（photons/view/mm² @1m，含 bowtie+平滤过）按能量加权算：

            Ka,r = Σ_E Φ_E·E·(μ_en/ρ)_air ， 参考点 = 等中心向源侧 15 mm
            DAP  = Ka(等中心面) × 该面照射野面积  [mGy·cm²]

        与 `estimate_run_dose`（介质内吸收剂量）/`estimate_ctdi`（体模内剂量指数）
        **并列但不可互换**：Ka,r 没体模、没散射、也没旋转累积（按帧 mAs 线性累加）。
        """
        if self._kerma_sim is None:
            phys = replace(self.physics, monochromatic_kev=None,
                           energy_count=max(20, int(self.dose_energy_bins)))
            self._kerma_sim = XcistSim(geometry=self.geometry, physics=phys,
                                       center_offset_mm=(0.0, 0.0, 0.0))
        e, net = self._kerma_sim.spectrum()
        n = int(n_frames if n_frames is not None else self.rotate_cfg.total_frame)
        mAs = self.physics.mA * self.physics.exposure_s * n
        out = estimate_air_kerma(
            e, net, sid_mm=self.geometry.sid, sdd_mm=self.geometry.sdd,
            n_col=self.geometry.n_col, n_row=self.geometry.n_row,
            col_size_mm=self.geometry.col_size, row_size_mm=self.geometry.row_size,
            n_frames=n, mAs_total=mAs, cell=cell)
        out["physics"] = {"spectrum_filename": self.physics.spectrum_filename,
                          "flat_filter_mm": list(self.physics.flat_filter_mm),
                          "mA": self.physics.mA,
                          "exposure_s": self.physics.exposure_s}
        out["n_frames_protocol"] = int(self.rotate_cfg.total_frame)
        return out

    def estimate_run_dose(self, roi_mask: np.ndarray | None = None,
                          verbose: bool = False) -> dict | None:
        """估本次旋转协议的吸收剂量；`dose_cfg is None` 时返回 None（无副作用）。

        做法：按协议取 `dose_subsample` 个均匀视角逐 view 沉积，再按
        总帧数/采样数线性缩放（每 view 剂量随角度平滑变化，子采样×缩放
        是无偏估计；`dose_subsample=1` 即全协议逐帧）。每 view 的束流 = 一帧
        曝光（`exposure_s`），总 mAs = mA × exposure_s × 总帧数。

        Returns:
            None 或 `{"dose": (N,N,S) mGy float32, "meta": {...}}`
        """
        if self.dose_cfg is None:
            return None
        dc = self.dose_cfg
        hu = self._dose_hu_ref if self._dose_hu_ref is not None else self._dose_hu_fallback
        hu_source = "reference_ct" if self._dose_hu_ref is not None else "mu_approx"
        if hu is None:
            raise RuntimeError(
                "已启用剂量估计但既未 set_dose_reference_hu(...)，也没有可回退的 μ 体积")
        n, ns = int(dc.n_voxel), int(self.dose_slices)
        if hu.shape[0] != n or hu.shape[1] != n:
            raise ValueError(
                f"剂量体积面内形状 {hu.shape[:2]} 应与 dose_cfg (n_voxel={n}) 一致；"
                "请用 `set_dose_reference_hu` 提供 (N,N,S) 的 HU 网格")
        hu_use = hu
        ns_use = int(hu_use.shape[2])

        sim = self._get_dose_sim()
        alphas = np.array([self._alpha_deg(f)
                           for f in range(self.rotate_cfg.total_frame)], dtype=float)
        betas = np.array([self.geometry.alpha_to_start_angle(a) for a in alphas])
        n_all = betas.size
        k = int(max(1, min(self.dose_subsample, n_all)))
        idx = np.unique(np.linspace(0, n_all - 1, k).round().astype(int))
        res = estimate_dose(sim._ct.cfg, hu_use, np.deg2rad(betas[idx]),
                            dose_cfg=dc, verbose=verbose)
        scale = n_all / float(idx.size)
        dose = (res["dose"] * scale).astype(np.float32)

        stats = {
            "whole_mean": float(dose.mean()),
            "whole_p99_9": max_dose_ref(dose),
            "whole_max": float(dose.max()),
            "iso": float(dose[n // 2, n // 2, ns_use // 2]),
        }
        if roi_mask is not None:
            m = np.asarray(roi_mask, dtype=bool)
            if m.shape == dose.shape and m.any():
                stats["roi_mean"] = float(dose[m].mean())
                stats["roi_p99_9"] = float(np.percentile(dose[m], 99.9))
        meta = {
            **res["meta"],
            "hu_source": hu_source,
            "n_views_total": int(n_all),
            "n_views_sampled": int(idx.size),
            "scale_factor": float(scale),
            "sampled_alpha_deg": [float(alphas[i]) for i in idx],
            "slices_used": int(ns_use),
            "mA": float(self.physics.mA),
            "exposure_s": float(self.physics.exposure_s),
            "mAs_total": float(self.physics.mA * self.physics.exposure_s * n_all),
            "spectrum_filename": self.physics.spectrum_filename,
            "flat_filter_mm": list(self.physics.flat_filter_mm),
            "stats": stats,
            "note": ("吸收剂量估计（被照体积内、累积整个旋转协议）；非 DAP/CTDI，"
                    "材料仅 water/bone（未含碘/金属），未含床移动/散射分离"),
        }
        # 额外给一个 CTDI 口径（标准化体模 + 360° + 100mm z 平均 + center/periph 加权）
        if self.compute_ctdi:
            try:
                ctdi = self.estimate_ctdi(verbose=verbose)
                meta["ctdi"] = {k: v for k, v in ctdi.items() if k != "dose"}
            except Exception as exc:            # noqa: BLE001
                meta["ctdi"] = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                meta["kerma"] = self.estimate_air_kerma()
            except Exception as exc:            # noqa: BLE001
                meta["kerma"] = {"error": f"{type(exc).__name__}: {exc}"}
        return {"dose": dose, "meta": meta}

    def _alpha_deg(self, frame: int) -> float:
        return float(self.rotate_cfg.get_rotation_angle_at_frame(frame)[0])

    def get_R_T_at_frame(self, frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        """与 TorchDRR 相同的 c2w R/T（复用同一几何公式，保证标签/深度对齐）。"""
        rotation = self.rotate_cfg.get_rotaiton_radian_at_frame(frame)
        rotations = torch.tensor([rotation])
        rot = convert(
            rotations,
            self.translations,
            parameterization=self.rotate_cfg.parameterization,
            convention=self.rotate_cfg.convention,
        )
        reorient = RigidTransform(self.reorient)
        pose = reorient.compose(rot)      # Rot @ reorient
        return pose.rotation, pose.translation

    # ------------------------------------------------------------------ phantom
    def _mono_phantom(self, volume) -> tuple[str, np.ndarray]:
        """μ 体积 → 单材料水当量密度体。

        返回 `(json 路径, 网格中心世界坐标)`；后者必须作为
        `cfg.phantom.centerOffset` 传给 gecatsim（体积 AABB 中心不等于等中心时，
        例如 reader 裁切出的体积，否则整个物体会被平移）。
        """
        mu = _as_numpy_3d(volume)
        affine = np.asarray(volume.affine, dtype=float)
        from ...xcist.materials import canonical_voxel_mm, world_aabb
        lo, hi = world_aabb(affine, mu.shape)
        center = (lo + hi) / 2.0
        half = np.maximum((hi - lo) / 2.0, 1e-3)
        voxel = self.grid_voxel_mm or canonical_voxel_mm(affine)
        grid = resample_scalar_to_canonical(mu, affine, center, half, voxel, order=1)
        self._maybe_attach_mu_fallback(grid, voxel, center, self.monochromatic_kev)
        mu_w = float(np.asarray(__import__("gecatsim").GetMu(
            "water", [self.monochromatic_kev])).ravel()[0]) / 10.0
        phantom = VoxelPhantom(arrays=mu_to_water_density(grid, mu_w),
                               voxel_mm=tuple(voxel), name=f"{self.name}_mono")
        return phantom.write(self.sim.phantom_out_dir), center

    def _ensure_poly_grid(self, dyn_volume) -> None:
        if self._grid is not None:
            return
        refs = [(aff, arr.shape) for arr, aff in self._static_layers]
        refs.append((np.asarray(dyn_volume.affine, dtype=float), _as_numpy_3d(dyn_volume).shape))
        self._grid = union_canonical_grid(refs, self.grid_voxel_mm)

    def _layer_materials_poly(self, arr, affine, coronary=None):
        center, half, voxel = self._grid
        grid = resample_scalar_to_canonical(arr, affine, center, half, voxel, order=1)
        mask = None
        if coronary is not None:
            cm = resample_scalar_to_canonical(
                (np.asarray(coronary) > 0).astype(np.float32), affine, center, half, voxel,
                order=0)
            mask = cm > 0.5
        return mu_to_materials(grid, coronary_mask=mask, reference_kev=self.reference_kev)

    def _build_static_mats(self) -> None:
        mats = []
        for arr, affine in self._static_layers:
            mats.append(self._layer_materials_poly(arr, affine, coronary=None))
        self._static_mats = sum_materials(*mats) if mats else {}

    # ------------------------------------------------------------------ 投影
    def get_projections_at_frames(self, frames, volume, coronary) -> torch.Tensor:
        n = len(frames)
        h, w = self.image_size
        # 新一轮 run 的开始：丢弃上一轮的 μ 回退体积
        self._dose_hu_fallback = None
        if self.mode == "poly":
            if self._dynamic_seen:
                # 新一轮 run：抛弃上一轮的静态层缓存
                self.reset()
            arr = _as_numpy_3d(volume)
            affine = np.asarray(volume.affine, dtype=float)
            if not any(arr.shape == a.shape and np.array_equal(affine, af)
                       for a, af in self._static_layers):
                self._static_layers.append((arr, affine))
            return torch.zeros(n, 1, h, w, dtype=torch.float32, device=self.device)

        # mono: 批量（均匀步进时一次多 view）或逐帧
        json, center = self._mono_phantom(volume)
        f0 = frames[0]
        contiguous = list(frames) == list(range(f0, f0 + n))
        if n == 1:
            e = self.sim.line_integral(self._alpha_deg(f0), json, center)
            return torch.from_numpy(e)[None, None].float()
        if contiguous:
            step = float(self.rotate_cfg.angular_velocity) / float(self.rotate_cfg.fps)
            e = self.sim.line_integral_multi(json, self._alpha_deg(f0), step, n, center)
            return torch.from_numpy(e)[:, None].float()
        out = np.stack([self.sim.line_integral(self._alpha_deg(f), json, center)
                        for f in frames])
        return torch.from_numpy(out)[:, None].float()

    def get_projection_at_frame(self, frame, volume, coronary) -> torch.Tensor:
        if frame == 0:
            self._dose_hu_fallback = None      # 新一轮 run 的第一帧
        if self.mode == "mono":
            json, center = self._mono_phantom(volume)
            e = self.sim.line_integral(self._alpha_deg(frame), json, center)
            return torch.from_numpy(e)[None, None].float()

        # poly: 延迟融合（静态层 + 本帧动态层）
        self._dynamic_seen = True
        self._ensure_poly_grid(volume)
        if self._static_mats is None:
            self._build_static_mats()
        arr = _as_numpy_3d(volume)
        corm = _as_numpy_3d(coronary) if coronary is not None else None
        dyn = self._layer_materials_poly(arr, np.asarray(volume.affine, dtype=float),
                                         coronary=corm)
        fused = sum_materials(self._static_mats, dyn)
        center, half, voxel = self._grid
        if self.dose_cfg is not None and self._dose_hu_ref is None \
                and self._dose_hu_fallback is None:
            mu_fused = np.zeros_like(next(iter(fused.values()))) if fused else None
            for mat, rho in fused.items():
                mu_fused += rho * mu_of_material_mm(mat, self.reference_kev)
            if mu_fused is not None:
                self._maybe_attach_mu_fallback(mu_fused, voxel, center, self.reference_kev)
        phantom = VoxelPhantom(arrays=fused, voxel_mm=tuple(voxel),
                               name=f"{self.name}_poly")
        json = phantom.write(self.sim.phantom_out_dir)
        t = self.sim.transmission(self._alpha_deg(frame), json, center)
        e = -np.log(np.clip(t, 1e-8, 1.0)).astype(np.float32)
        return torch.from_numpy(e)[None, None].float()

    # ------------------------------------------------------------------ 其他
    def reset(self) -> None:
        """清空 poly 静态层缓存（同一实例复用于多次 run 时调用）。"""
        self._static_layers = []
        self._static_mats = None
        self._grid = None
        self._dynamic_seen = False
        self._dose_hu_fallback = None
