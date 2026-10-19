"""gecatsim CatSim 封装：单/多 view 平板投影。

对外能力：
- 用 XcistGeometry + VoxelPhantom 运行 1..N view 的 phantom scan；
- 缓存 air scan，输出逐帧透射/强度图像 (row, col)；
- 暴露 center_offset / 翻转 / alpha_to_beta 等标定参数。

物理默认（用户给定 90kV / 50mA / 0.1mmCu / 3.3ms，平板，无 bowtie）；
量子/电子噪声默认关闭（spike 便于与 DiffDRR 对比，可开）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

import gecatsim as xc

from .geometry import XcistGeometry

#: gecatsim 包内 phantom 目录（写 .vp/.raw 的默认位置，可被 my_path 找到）
_GC_PACKAGE = os.path.join(
    "/media/data3/sj/Code/Gen4D-dev-accelerate",
    ".pixi/envs/default/lib/python3.14/site-packages/gecatsim",
)
GC_PHANTOM_DIR = os.path.join(_GC_PACKAGE, "phantom", "gen4d")

# 标定参数初值（待真实数据对比后固化到 geometry.py）
#: `cfg.phantom.centerOffset` 的默认值：体模网格**中心**在世界中的位置(mm)。
#: 缺省 (0,0,0) = 网格中心在等中心；体积 AABB 中心偏离等中心时必须显式传入。
GRID_CENTER_OFFSET_MM = (0.0, 0.0, 0.0)


@dataclass
class XcistPhysics:
    spectrum_filename: str = "tungsten_tar7.0_90_filt.dat"
    flat_filter_mm: list = field(default_factory=lambda: ["cu", 0.1])
    mA: float = 50.0
    exposure_s: float = 3.3e-3           # 每帧曝光（beam-on）时长
    fps: float = 60.0                    # 帧率（决定帧间隔与 dutyRatio）
    monochromatic_kev: float | None = None  # 非空→单能模式（与 DiffDRR 线积分同构）
    enable_quantum_noise: bool = False
    enable_electronic_noise: bool = False
    col_sample: int = 1
    row_sample: int = 1
    src_x_sample: int = 1
    src_y_sample: int = 1
    view_sample: int = 1
    energy_count: int = 20

    @property
    def frame_interval_s(self) -> float:
        return 1.0 / float(self.fps)

    @property
    def duty_ratio(self) -> float:
        """束流开启占空比 = 曝光时长 / 帧间隔（60fps+3.3ms → ≈0.198）。"""
        return min(max(self.exposure_s / self.frame_interval_s, 0.0), 1.0)


class XcistSim:
    """单进程内复用同一个 CatSim 实例跑多帧。"""

    def __init__(
        self,
        geometry: XcistGeometry,
        physics: XcistPhysics | None = None,
        phantom_out_dir: str = GC_PHANTOM_DIR,
        center_offset_mm: tuple[float, float, float] = GRID_CENTER_OFFSET_MM,
        verbose: bool = False,
    ):
        self.geometry = geometry
        self.physics = physics or XcistPhysics()
        self.phantom_out_dir = phantom_out_dir or GC_PHANTOM_DIR
        self.center_offset_mm = list(center_offset_mm)
        self.verbose = verbose
        os.makedirs(self.phantom_out_dir, exist_ok=True)
        self._ct = xc.CatSim()
        self._configure()
        self._air = None            # 缓存的 air 图（无体模时均匀）
        self._air_key = None        # air 缓存对应的协议签名
        self._spec = None           # 缓存的滤过谱 (Evec, netIvec)

    # ------------------------------------------------------------------ cfg
    def _configure(self) -> None:
        ct = self._ct
        p = self.physics
        ct.resultsName = "/tmp/gen4d_xcist/out"
        os.makedirs("/tmp/gen4d_xcist", exist_ok=True)
        ct.protocol.scanTypes = [1, 0, 1, 0]    # air + phantom（无 offset/prep）
        # 物理
        ct.protocol.spectrumFilename = p.spectrum_filename
        ct.protocol.flatFilter = list(p.flat_filter_mm)
        ct.protocol.mA = float(p.mA)
        ct.protocol.bowtie = []                  # C-arm 平板：无 bowtie
        # 曝光：gecatsim 中 beam-on = viewTime*dutyRatio/subViewCount，
        #   viewTime = rotationTime/viewsPerRotation；而 gantry 角 = startAngle + cfg.time/rotationTime*360，
        #   其中 cfg.time 含 -0.5*viewTime 偏移。若把 rotationTime 设为帧间隔并令 viewsPerRotation=1，
        #   该偏移会使实际视角整体偏移约 -144°（破坏几何标定）。
        # 因此单 view 采用与"真实协议每帧 3.3ms 束流"完全等价且在几何上已验证的计时：
        #   rotationTime=exposure_s, viewsPerRotation=1, dutyRatio=1 → beam-on 恰为 exposure_s 且 cfg.time=0。
        # 真实 fps/角速度/时长等由 XcistScanProtocol 描述并写入元数据（不影响单帧通量）。
        ct.protocol.viewsPerRotation = 1
        ct.protocol.viewCount = 1
        ct.protocol.startViewId = 0
        ct.protocol.stopViewId = 0
        ct.protocol.rotationTime = float(p.exposure_s)
        ct.protocol.dutyRatio = 1.0
        ct.protocol.rotationDirection = 1
        # 单能模式：xcist 输出 T=exp(-∫μds)，可当作线积分投影（与 DiffDRR 同构）
        if p.monochromatic_kev is not None:
            ct.physics.monochromatic = float(p.monochromatic_kev)
            ct.physics.energyCount = 1
        # 噪声 / 采样
        ct.physics.enableQuantumNoise = int(p.enable_quantum_noise)
        ct.physics.enableElectronicNoise = int(p.enable_electronic_noise)
        if p.monochromatic_kev is None:
            ct.physics.energyCount = int(p.energy_count)
        ct.physics.colSampleCount = p.col_sample
        ct.physics.rowSampleCount = p.row_sample
        ct.physics.srcXSampleCount = p.src_x_sample
        ct.physics.srcYSampleCount = p.src_y_sample
        ct.physics.viewSampleCount = p.view_sample
        # 体模：voxelized
        ct.phantom.callback = "Phantom_Voxelized"
        ct.phantom.projectorCallback = "C_Projector_Voxelized"
        ct.phantom.centerOffset = self.center_offset_mm
        # 几何
        self.geometry.apply_to_cfg(ct.cfg)

    # ------------------------------------------------------------------ 运行
    def _apply_single_view_protocol(self, alpha_deg: float) -> None:
        """单 view 协议：beam-on=exposure_s 且 cfg.time=0（几何已验证）。"""
        ct = self._ct
        ct.protocol.rotationTime = float(self.physics.exposure_s)
        ct.protocol.dutyRatio = 1.0
        self.geometry.apply_frame_to_cfg(ct.cfg, alpha_deg)

    def _run_scan(self, scan_type: str, alpha_deg: float) -> None:
        ct = self._ct
        self._apply_single_view_protocol(alpha_deg)
        if scan_type == "air":
            ct.protocol.scanTypes = [1, 0, 0, 0]
            ct.air_scan()
            self._last_ext = ".air"
        else:
            ct.protocol.scanTypes = [0, 0, 1, 0]
            ct.phantom_scan()
            self._last_ext = ".scan"

    def _read_image(self, n_views: int = 1) -> np.ndarray:
        ct = self._ct
        img = xc.rawread(ct.resultsName + self._last_ext,
                         [n_views, ct.scanner.detectorRowCount,
                          ct.scanner.detectorColCount], "float")
        img = img.astype(np.float32)
        # 与 DiffDRR 显示方向对齐（标定：需 180° 旋转，见 geometry.FLIP_*）
        if self.geometry.flip_row < 0:
            img = img[..., ::-1, :]
        if self.geometry.flip_col < 0:
            img = img[..., :, ::-1]
        return img[0] if n_views == 1 else img

    def _air_sig(self):
        """air 计数与协议（束流时间）相关，缓存必须按协议签名失效。"""
        c = self._ct.protocol
        return (round(float(c.rotationTime), 12), round(float(c.dutyRatio), 12),
                round(float(c.viewsPerRotation), 9),
                int(self._ct.physics.viewSampleCount))

    def _air_now(self) -> np.ndarray:
        ct = self._ct
        ct.protocol.scanTypes = [1, 0, 0, 0]
        ct.air_scan()
        self._last_ext = ".air"
        return self._read_image(1)

    def air(self, alpha_deg: float = 0.0, reuse: bool = True) -> np.ndarray:
        """无体模的入射图 [row,col]（按协议缓存；无 bowtie 时与角度无关）。"""
        key = self._air_sig()
        if reuse and self._air is not None and self._air_key == key:
            return self._air
        air = self._air_now()
        if reuse:
            self._air = air
            self._air_key = key
        return air

    def spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """滤过后的**每 cell 注量谱**：`(Evec[keV], netIvec[cells, nE])`。

        单位 = **photons/view/mm² at 1 m**（已含 mA×viewTime、bowtie 与平滤过），
        与 CatSim 内部 `cfg.spec.netIvec` 完全一致。取法：跑一次无体模 air scan
        （CatSim 在扫描准备阶段就会算好该谱），结果按当前 physics 缓存。
        供 `dose.estimate_air_kerma`（空气比释动能 / DAP）使用。
        """
        if self._spec is not None:
            return self._spec
        self.air()                              # 无体模，填充 cfg.det / cfg.spec
        spec = self._ct.cfg.spec
        e = np.asarray(spec.Evec, dtype=float).ravel()
        net = np.asarray(spec.netIvec, dtype=float)
        if net.ndim < 2:
            net = net[None, :]
        self._spec = (e, net.reshape(-1, e.size))
        return self._spec

    def _apply_center_offset(self, center_offset_mm) -> None:
        """设置 `cfg.phantom.centerOffset`。

        语义（已实测）：体模网格**中心**所在的世界坐标（mm）；网格尺寸/体素由 .vp 给出。
        即 `VoxelPhantom.write` 写出的网格会被摆到 `centerOffset` 处，缺省 (0,0,0)
        → 网格中心 = 等中心。当体积的 AABB 中心不在等中心时（例如 reader 裁切出
        的体积）必须传入其中心，否则整个物体会被平移（实测 NCC 0.42 → 0.9998）。
        """
        off = self.center_offset_mm if center_offset_mm is None else list(center_offset_mm)
        cur = list(getattr(self._ct.phantom, "centerOffset", []))
        if cur != list(off):
            self._ct.phantom.centerOffset = list(off)

    def project(self, alpha_deg: float, phantom_json: str,
                center_offset_mm=None) -> np.ndarray:
        """单帧 phantom scan 的探测器计数 [row,col]（含或不含噪声）。"""
        self._apply_center_offset(center_offset_mm)
        self._ct.phantom.filename = phantom_json
        self._run_scan("phantom", alpha_deg)
        return self._read_image()

    def transmission(self, alpha_deg: float, phantom_json: str,
                     center_offset_mm=None) -> np.ndarray:
        """单帧透射 T=scan/air（≈ exp(-E_eff)，用于与 DiffDRR 对比/后处理）。"""
        scan = self.project(alpha_deg, phantom_json, center_offset_mm)
        air = self.air(alpha_deg)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(air > 0, scan / air, 0.0)
        return np.clip(t, 0.0, 1.0)

    # -------------------------------------------------- 批量（多 view 一次仿真）
    def project_multi(self, phantom_json: str, alpha0_deg: float,
                      step_deg: float, n_views: int,
                      center_offset_mm=None) -> np.ndarray:
        """一次 CatSim 跑 n_views 个均匀角度视图，返回 (N,row,col) 计数。

        视角：beta(v) = startAngle + (-0.5+0.5·duty)·step + v·step（由
        OneScan 的 time 推进与 Gantry_Helical 推出），因此这里反算 startAngle
        抵消常数偏移；每 view 束流时间 = exposure_s。
        """
        ct = self._ct
        p = self.physics
        g = self.geometry
        view_time = p.frame_interval_s
        vpr = 360.0 / float(step_deg)
        duty = min(max(p.exposure_s / view_time, 0.0), 1.0)
        self._apply_center_offset(center_offset_mm)
        ct.phantom.filename = phantom_json
        ct.protocol.scanTypes = [0, 0, 1, 0]
        ct.protocol.viewsPerRotation = float(vpr)
        ct.protocol.rotationTime = float(view_time * vpr)
        ct.protocol.dutyRatio = float(duty)
        ct.protocol.viewCount = int(n_views)
        ct.protocol.startViewId = 0
        ct.protocol.stopViewId = int(n_views) - 1
        ct.protocol.rotationDirection = int(g.rotation_direction)
        delta = (-0.5 + 0.5 * duty) * float(step_deg) * g.rotation_direction
        ct.protocol.startAngle = float(g.alpha_to_start_angle(alpha0_deg) - delta)
        ct.phantom_scan()
        self._last_ext = ".scan"
        return self._read_image(n_views)

    def transmission_multi(self, phantom_json: str, alpha0_deg: float,
                           step_deg: float, n_views: int,
                           center_offset_mm=None) -> np.ndarray:
        scan = self.project_multi(phantom_json, alpha0_deg, step_deg, n_views,
                                  center_offset_mm)
        air = self.air(alpha0_deg)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(air[None] > 0, scan / air[None], 0.0)
        return np.clip(t, 0.0, 1.0)

    def line_integral(self, alpha_deg: float, phantom_json: str,
                      center_offset_mm=None) -> np.ndarray:
        """单帧线积分 E=-ln(T)（单能模式下即 ∫μds，可直接给 RotateDSA 用）。"""
        t = self.transmission(alpha_deg, phantom_json, center_offset_mm)
        return -np.log(np.clip(t, 1e-8, 1.0)).astype(np.float32)

    def line_integral_multi(self, phantom_json: str, alpha0_deg: float,
                            step_deg: float, n_views: int,
                            center_offset_mm=None) -> np.ndarray:
        t = self.transmission_multi(phantom_json, alpha0_deg, step_deg, n_views,
                                    center_offset_mm)
        return -np.log(np.clip(t, 1e-8, 1.0)).astype(np.float32)

    # ------------------------------------------------------------------ 便捷
    def set_center_offset(self, offset_mm) -> None:
        self.center_offset_mm = list(offset_mm)
        self._ct.phantom.centerOffset = self.center_offset_mm

    @property
    def detector_size(self) -> tuple[int, int]:
        return (self._ct.scanner.detectorRowCount, self._ct.scanner.detectorColCount)
