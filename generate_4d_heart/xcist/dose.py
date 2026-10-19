"""XCIST 剂量估计（per-voxel absorbed dose, mGy）——适配我们的 C-arm 平板几何。

XCIST 的剂量重建 `catdoserecon` 从一张 CT 图像（作为剂量沉积介质）+ 几何/能谱/
mAs/视角协议，逐能量 bin 用 C 库 (`DD3Dose`) 追迹射线、沉积能量，最终按体素
质量换算成吸收剂量 (mGy)。

本模块相对官方实现的差异/修补：
  1. 官方 `img2vol` 依赖已废弃的 `scipy.interpolate.interp2d` 且在 numpy 下
     有 2D/3D 布尔掩码 bug → 这里自行完成 HU→μ→密度→质量 的转换（公式与官方一致）。
  2. 官方视角由 protocol 推导、**从 0 开始均匀**，无法表达任意起始角的 C-arm
     协议 → 这里支持**显式 viewangles 数组**（radians）。
  3. 复用我们的 `XcistGeometry`（sid/sdd/平板/AP-PA）与 `XcistPhysics`（能谱/过滤/mAs）。

单位与网格：
  - 输入 volume 轴序 (x, y, z) = (nVoxel, nVoxel, sliceCount)，世界为
    iso 原点、X 右 / Y 前(源侧) / Z 头向（与 phantom.py / geometry.py 一致）。
  - 输出 dose (nVoxel, nVoxel, sliceCount) 单位 mGy（整个扫描/所有 view 的沉积剂量）。

主要限制（review 注意）：
  - 材料仅 water/bone 二分类（阈值 `water_threshold`，μ 单位 1/cm），
    密度由 μ 反推；碘/金属未单独建模。
  - 剂量是"在该 CT 体上按给定几何/能谱/mAs 沉积"的估计，不含床位移动(zshifts)、
    散射剂量分离等；do_convol 使用官方卷积核改善分布。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from gecatsim.pyfiles.CommonTools import feval
from gecatsim.dose.pyfiles.calcDetectorFlux import calcDetectorFlux
from gecatsim.dose.pyfiles.C_DD3Dose import C_DD3Dose
from gecatsim.dose.pyfiles.DoseConv import DoseConv
from gecatsim.dose.pyfiles.catdoserecon import (
    convert_to_mGy, WriteDoseFiles, myDC, load_C_lib,
)

from .geometry import XcistGeometry
from .sim import XcistPhysics

#: 传给 gecatsim 射线追迹代码的体素数组，其**第 2 轴（rows/深度）= 世界 -Y**。
#:
#: 与 `.vp` 体模同一约定（见 phantom.FLIP_Y_IN_VP）。实测铁证
#: （`temp/xcist_dose_axis.py`，α=0 源在 +Y、平方反比）：同一水球放在数组
#: j=+40mm 得峰值剂量 1.644、j=-40mm 得 2.031 → 更近源的一侧是 j 负向。
#: α=90（源在 -X）同法验证：数组 i 正方向 = 世界 +X（X 不镜像）。
FLIP_VOL_Y_IN_DOSE = True


def install_interp2d_shim() -> None:
    """给 scipy>=1.14 提供 interp2d 兼容实现（官方 img2vol 可能仍被调用）。"""
    import scipy.interpolate as si
    if getattr(si, "_gen4d_shim", False):
        return
    def _shim(x, y, z, kind="linear", **kw):
        from scipy.interpolate import RegularGridInterpolator
        xi = np.asarray(x, float); yi = np.asarray(y, float)
        rgi = RegularGridInterpolator((yi, xi), np.asarray(z, float),
                                      method="linear", bounds_error=False,
                                      fill_value=0.0)
        def call(XI, YI):
            XI = np.atleast_1d(np.asarray(XI, float)); YI = np.atleast_1d(np.asarray(YI, float))
            XX, YY = np.meshgrid(XI, YI, indexing="xy")
            pts = np.stack([YY.ravel(), XX.ravel()], axis=-1)
            return rgi(pts).reshape(len(YI), len(XI))
        return call
    si.interp2d = _shim
    si._gen4d_shim = True


@dataclass
class DoseConfig:
    n_voxel: int = 128
    fov_mm: float = 240.0
    slice_thickness_mm: float = 1.0
    # HU→μ 的标定（与主管线一致：μ_cm = (HU-huOffset)/1000*mu_mm*10）
    hu_offset: float = -1000.0
    mu_mm_water: float = 0.02     # 1/mm @1 g/cc
    # 剂量重建材料参数（官方默认）
    mu_water_cm: float = 0.2      # 1/cm
    mu_bone_cm: float = 0.5530    # 1/cm
    bone_density: float = 1.92    # g/cc
    water_threshold: float = 0.25 # μ(1/cm) 分骨/水
    air_threshold: float = 0.04
    outer_air_threshold: float = 0.04
    do_convol: bool = False
    convert_to_mGy: bool = True


def hu_to_dose_basis(hu_xyz: np.ndarray, cfg: DoseConfig, assume_hu: bool | None = None):
    """HU (x,y,z)=(N,N,S) → (vol μ[1/cm], dens[g/cc], mass[g/voxel])，仿官方 img2vol。

    `assume_hu`：``None`` 保持官方启发式（max>10 视为 HU，否则视为已是 μ）；
    ``True`` 强制按 HU 解释（均匀水/PMAA 体模这类 max≤10 的输入必须显式指定）。

    注意：官方 `img2vol` 用 **各向同性** 体素 `(fov/n)^3` 算质量，隐含
    `sliceThickness == fov/n`；而能量沉积用的是 `(voxel_xy, voxel_z)`。
    两者不等时剂量会**正比于层厚**（实测 2/4/5/8mm → 剂量 1:2:2.5:4，纯假象）。
    这里改为真实体素体积 `vxy² · vz`，剂量因此与层厚无关（正确物理）。
    """
    vol = np.asarray(hu_xyz, dtype=np.float32).copy()
    if vol.shape[0] != cfg.n_voxel or vol.shape[1] != cfg.n_voxel:
        raise ValueError(
            f"HU 体面内形状 {vol.shape[:2]} 应与 DoseConfig.n_voxel={cfg.n_voxel} 一致")
    if assume_hu is True or (assume_hu is None and np.max(vol) > 10):
        vol = (vol - cfg.hu_offset) / 1000.0 * cfg.mu_mm_water * 10.0  # 1/cm
    N = cfg.n_voxel
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.sqrt((xx - (N - 1) / 2) ** 2 + (yy - (N - 1) / 2) ** 2)
    outer = r > (N / 2)                                     # 内切圆外→0
    mask_air = (vol < cfg.outer_air_threshold) | outer[..., None]
    vol[mask_air] = 0.0
    vol[np.isnan(vol) | np.isinf(vol)] = 0.0
    mask_bone = vol > cfg.water_threshold
    dens = vol / (np.where(mask_bone, cfg.mu_bone_cm / cfg.bone_density,
                           cfg.mu_water_cm))
    vxy_cm = cfg.fov_mm / 10.0 / cfg.n_voxel
    vz_cm = float(cfg.slice_thickness_mm) / 10.0
    mass = dens * (vxy_cm ** 2) * vz_cm                    # g（真实体素体积）
    return vol.astype(np.float32), dens.astype(np.float32), mass.astype(np.float32)


def _detector_and_source(cfg):
    """从 cfg 取探测器像素坐标与源位置（复刻官方 catdoserecon 的取法）。"""
    import numpy.matlib
    cfg = feval(cfg.scanner.detectorCallback, cfg)
    nrdetcols = cfg.scanner.detectorColCount
    nrdetrows = cfg.scanner.detectorRowCount
    det_xyz = np.zeros((3, cfg.det.totalNumCells))
    for m in np.arange(cfg.det.nMod):
        n_cells = cfg.det.nCells
        cell_ind = np.arange(cfg.det.startIndices[m], cfg.det.startIndices[m] + n_cells)
        det_xyz[:, cell_ind] = np.matlib.repmat(cfg.det.modCoords[m], n_cells, 1).T + \
            np.array([cfg.det.uvecs[m], cfg.det.vvecs[m]]).T @ cfg.det.cellCoords[np.arange(n_cells)].T
    det_xyz = np.reshape(det_xyz, (3, nrdetrows, nrdetcols), order='F').astype(np.single)
    det_x = np.squeeze(det_xyz[0, int(np.ceil(nrdetrows / 2)), :])
    det_y = np.squeeze(det_xyz[1, int(np.ceil(nrdetrows / 2)), :])
    det_z = np.squeeze(det_xyz[2, :, int(np.ceil(nrdetcols / 2))])
    cfg = feval(cfg.scanner.focalspotCallback, cfg)
    src = np.single(np.mean(cfg.src.samples, 0))
    cfg = feval(cfg.physics.rayAngleCallback, cfg)
    return cfg, (float(src[0]), float(src[1]), float(src[2])), (nrdetcols, nrdetrows), det_x, det_y, det_z


def estimate_dose(
    sim_cfg,
    hu_xyz: np.ndarray,
    viewangles_rad: Sequence[float],
    *,
    dose_cfg: DoseConfig | None = None,
    dose_out: Optional[str] = None,
    verbose: bool = True,
    assume_hu: bool | None = None,
):
    """在给定 CatSim cfg 上估计吸收剂量（显式视角）。

    Args:
        sim_cfg: 一个已配置好的 gecatsim cfg（含 scanner/protocol/physics/spectrum），
            通常来自 `XcistSim(...)._ct.cfg`；需先设置好 mA / rotationTime /
            viewsPerRotation / dutyRatio（决定每 view 通量）与 energyCount。
        hu_xyz: (N,N,S) HU 体（世界 = iso 原点，X 右/Y 前/Z 头向）。
        viewangles_rad: 各 view 的 gantry 旋转角（radians，绕 +Z）。
        dose_cfg: 剂量重建参数。
        dose_out: 若给出，写 `<dose_out>.mGy`。
    Returns:
        dict( dose=(N,N,S) mGy, meta={...} )
    """
    install_interp2d_shim()
    cfg = sim_cfg
    dc = dose_cfg or DoseConfig()
    hu_xyz = np.asarray(hu_xyz, dtype=np.float32)
    N = hu_xyz.shape[0]
    nrplanes = hu_xyz.shape[2]
    # 体素第 2 轴 = 世界 -Y → 送入前翻转，保证剂量沉积在正确解剖位置
    hu_in = (np.ascontiguousarray(hu_xyz[:, ::-1, :]) if FLIP_VOL_Y_IN_DOSE
             else np.ascontiguousarray(hu_xyz))
    vol, dens_vol, mass_vol = hu_to_dose_basis(hu_in, dc, assume_hu=assume_hu)

    cfg, (x0, y0, z0), (nrdetcols, nrdetrows), det_x, det_y, det_z = _detector_and_source(cfg)

    # 每 view 通量 → sino[E, cells, views]
    cfg.sim.subViewCount = 1
    cfg.sim.isOffsetScan = 0
    cfg = feval(cfg.protocol.spectrumCallback, cfg)
    viewangles = np.asarray(viewangles_rad, dtype=np.single)
    nrviews = int(viewangles.size)
    sino = np.single(np.zeros((cfg.physics.energyCount, cfg.det.totalNumCells, nrviews)))
    for vi in range(nrviews):
        sino[:, :, vi] = calcDetectorFlux(vi, vi, cfg).T
    sino = sino * cfg.scanner.detectorColSize * cfg.scanner.detectorRowSize / cfg.det.activeArea
    zshifts = np.zeros(nrviews, dtype=np.single)

    cfg.dose.doselib = load_C_lib()

    voxel_xy = dc.fov_mm / dc.n_voxel
    voxel_z = dc.slice_thickness_mm
    ind_bone = vol > dc.water_threshold
    ind_water = np.logical_and(vol < dc.water_threshold, vol > dc.air_threshold)
    mask_highZ = np.single(ind_bone)

    dosevol = np.single(np.zeros_like(vol))
    mydc = myDC()
    if np.ndim(cfg.spec.Evec) < 1:
        cfg.spec.Evec = np.expand_dims(cfg.spec.Evec, 0)
    for e in range(cfg.physics.energyCount):
        if verbose:
            print(f"  [dose] energy bin {e + 1}/{cfg.physics.energyCount}")
        ee = [cfg.spec.Evec[e]]
        mu_water = float(np.asarray(__import__("gecatsim").GetMu("water", ee)).ravel()[0])
        this_vol = np.copy(vol)
        this_vol[ind_bone] = this_vol[ind_bone] * float(np.asarray(__import__("gecatsim").GetMu("bone", ee)).ravel()[0]) / dc.mu_bone_cm
        this_vol[ind_water] = this_vol[ind_water] * float(np.asarray(__import__("gecatsim").GetMu("water", ee)).ravel()[0]) / dc.mu_water_cm
        this_sino = np.reshape(sino[e], [cfg.det.totalNumCells, nrviews], order="F")
        this_dose, _ = C_DD3Dose(
            cfg, x0, y0, z0, np.int32(nrdetcols), np.int32(nrdetrows),
            det_x, det_y, det_z,
            np.single(0.0), np.single(0.0), np.single(0.0),
            np.single(viewangles), zshifts, np.int32(nrviews),
            this_sino, np.int32(N), np.int32(N), np.int32(nrplanes),
            this_vol, np.zeros_like(vol), np.single(voxel_xy), np.single(voxel_z),
        )
        if dc.do_convol:
            this_dose = DoseConv(cfg, mydc, ee, this_dose, this_vol, voxel_xy, mask_highZ)
        this_dose = this_dose * ee
        if dc.convert_to_mGy:
            this_dose = convert_to_mGy(this_dose, mass_vol)
        dosevol = dosevol + this_dose

    meta = {"n_voxel": N, "slices": nrplanes, "fov_mm": dc.fov_mm,
            "slice_thickness_mm": dc.slice_thickness_mm,
            "n_views": nrviews, "energy_bins": cfg.physics.energyCount,
            "unit": "mGy" if dc.convert_to_mGy else "keV",
            "flip_vol_y": bool(FLIP_VOL_Y_IN_DOSE)}
    # 翻回规范约定（第 2 轴 → 世界 +Y），与投影/标签/CT 解剖对齐
    dosevol = np.ascontiguousarray(dosevol[:, ::-1, :]) if FLIP_VOL_Y_IN_DOSE else dosevol
    if dose_out:
        cfg.dose.doseFileName = os.path.splitext(dose_out)[0]
        if not hasattr(cfg.dose, "convertTomGy"):
            cfg.dose.convertTomGy = int(dc.convert_to_mGy)
        cfg.dose.convertTomGy = int(dc.convert_to_mGy)
        WriteDoseFiles(cfg, "", dosevol, 1)
    return {"dose": np.asarray(dosevol), "meta": meta}


def max_dose_ref(dose: np.ndarray) -> float:
    """取百分比最大值（去除个别热点噪声），便于报告。"""
    d = np.asarray(dose).ravel()
    return float(np.percentile(d, 99.9))


# --------------------------------------------------------------------------- CTDI 口径
@dataclass
class CtdiConfig:
    """CTDI-like 估算口径（把"单体素患者剂量"换成可跨系统比较的剂量指数）。"""

    phantom_diameter_mm: float = 160.0   # 默认 16 cm 头模（32 cm 体模的周边孔在本平板扇束外，见 warnings）
    phantom_length_mm: float = 200.0     # 圆柱长度（需 > integration_mm 才有完整中央层）
    n_voxel: int = 64                    # 面内体素数（fov = phantom_diameter_mm）
    slice_thickness_mm: float = 2.0
    integration_mm: float = 100.0        # CTDI100 的 z 积分长度
    periph_radius_mm: float | None = None  # 周边测量点半径；None → 直径/2 - 10（16cm≈70，32cm≈150）
    view_density_per_360: float = 1.5    # 每度视角数（由协议推算：帧数/扫角度数）
    min_views: int = 36                  # 360° 最少视角数（过低会因角度采样不足而偏小）
    max_views: int = 72                  # 360° 最大实际沉积视角数（超出则按比例缩放）
    detector_downsample: int = 3          # 平板像素降采样倍数（保持面板尺寸/总通量，仅粗化射线采样）
    material_hu: float = 120.0           # PMMA ≈ +120（水当量可给 0，需 assume_hu=True）
    do_convol: bool = False

    def n_views(self) -> int:
        return int(max(self.min_views, round(360.0 * self.view_density_per_360)))


def estimate_ctdi_like(
    sim_cfg,
    ctdi_cfg: CtdiConfig | None = None,
    *,
    mAs_per_rotation: float | None = None,
    beam_width_mm: float | None = None,
    return_dose: bool = False,
    verbose: bool = False,
):
    """按 **CTDI 口径** 估算剂量指数（均匀水当量圆柱体模 + 360° 单圈 + z 平均）。

    与 `estimate_dose`（患者真实解剖、单点/ROI）的区别，正是 CTDI 的标准化之处：

    * 介质：**均匀水当量圆柱体模**（默认直径 16 cm = 头模/CTDI_16；32 cm 体模可显式指定，
      但周边孔 r=150 mm 常超出本平板在等中心的扇束宽度，见返回里的 `warnings`），而非患者 CT；
    * 旋转：**360° 全圈、静止床**（剂量按本协议推得的**每度视角数**采样，
      即保持与真实协议相同的角度密度，只是把 225° 补成 360°）；
    * 采样：**沿 z 平均 100 mm**（等价 100 mm 笔形电离室），得到 center / 4×periph；
    * 加权：``CTDI_w = 1/3·CTDI_center + 2/3·CTDI_periph``（periph 为 4 点平均）；
    * 归一：给出 **每 100 mAs** 的版本，便于与不同协议/机型的公开数据对照。

    返回同时给两套口径（见 `definitions`）：
    - ``ctdi100``：严格定义 ``∫_{-50}^{50} D dz / nT``（nT = 本几何在等中心的**束宽**），
      只有 ``nT ≲ 100 mm`` 时与临床 CTDI 同量级；
    - ``mean100``：宽束等价（中央 100 mm 的平均吸收剂量，不除 nT），
      对平板 C 臂这类"束宽 >> 100 mm"的几何更直观，**应优先引用** ``ctdi_w_mean100``。

    注意：本引擎**不含散射**，且 CTDI 标准测量用的是电离室 + bowtie，
    因此数值只能与"同类平板 C 臂/锥束 CT"做同口径比较，不能与医用 CT 的
    CTDIvol 直接等同。
    """
    c = ctdi_cfg or CtdiConfig()
    n = int(c.n_voxel)
    diam = float(c.phantom_diameter_mm)
    thick = float(c.slice_thickness_mm)
    ns = int(max(8, round(float(c.phantom_length_mm) / thick)))

    # 均匀圆柱体模（HU：内部 = material_hu，外部 = 空气 -1000）
    vxy = diam / n
    xv = (np.arange(n) - (n - 1) / 2.0) * vxy
    X, Y = np.meshgrid(xv, xv, indexing="ij")          # hu_xyz 轴序 (x, y, z)
    r_xy = np.hypot(X, Y)
    hu = np.broadcast_to(np.where(r_xy <= diam / 2.0, float(c.material_hu), -1000.0)[..., None],
                         (n, n, ns)).copy()

    dc = DoseConfig(n_voxel=n, fov_mm=diam, slice_thickness_mm=thick,
                    do_convol=bool(c.do_convol))
    n_views = c.n_views()                              # 协议等效的 360° 视角数
    n_used = int(min(n_views, max(1, int(c.max_views))))
    viewangles = np.deg2rad(np.linspace(0.0, 360.0, n_used, endpoint=False))
    res = estimate_dose(sim_cfg, hu, viewangles, dose_cfg=dc, verbose=verbose,
                        assume_hu=True)
    dose = np.asarray(res["dose"], dtype=np.float32)
    if n_used != n_views:                              # 视角降采样 → 按比例还原
        dose = dose * (n_views / float(n_used))
    n_views_eff = n_views

    # 中央 integration_mm 层厚的 slab（等价 100 mm 电离室）
    k = int(max(1, min(ns, round(float(c.integration_mm) / thick))))
    z0 = (ns - k) // 2
    slab = dose[:, :, z0:z0 + k]

    core_r = max(2.0, 2.0 * vxy)                       # 中心取样半径（≥2 体素）
    center_mask = r_xy <= core_r
    d_center = slab[center_mask].mean()

    # 4 个周边点（±x, ±y），各自小邻域平均
    rp = (float(c.periph_radius_mm) if c.periph_radius_mm is not None
          else max(5.0, diam / 2.0 - 10.0))
    periph_vals = []
    for ang in (0.0, 90.0, 180.0, 270.0):
        px, py = rp * np.cos(np.deg2rad(ang)), rp * np.sin(np.deg2rad(ang))
        m = np.hypot(X - px, Y - py) <= core_r
        if m.any():
            periph_vals.append(float(slab[m].mean()))
    d_periph = float(np.mean(periph_vals)) if periph_vals else float("nan")

    # 束宽 nT（等中心处）：探测器行覆盖 × 放大率倒数
    if beam_width_mm is None:
        try:
            s = sim_cfg.scanner
            beam_width_mm = (float(s.detectorRowCount) * float(s.detectorRowSize)
                             * float(s.sid) / float(s.sdd))
        except Exception:                              # noqa: BLE001
            beam_width_mm = float("nan")
    nT = float(beam_width_mm)

    mean100 = {"center": float(d_center), "periph": float(d_periph)}
    ctdi100 = {k2: (v * float(c.integration_mm) / nT if np.isfinite(nT) and nT > 0 else float("nan"))
               for k2, v in mean100.items()}
    ctdi_w_mean = (mean100["center"] + 2.0 * mean100["periph"]) / 3.0
    ctdi_w = (ctdi100["center"] + 2.0 * ctdi100["periph"]) / 3.0

    # 扇束横向覆盖诊断：平板在等中心只能覆盖 fan_fov_mm 宽，
    # 超过半个扇宽的周边点只在部分角度上被照射 → 周边剂量被低估。
    warn = []
    fan_fov = float("nan")
    cov = float("nan")
    try:
        s = sim_cfg.scanner
        fan_fov = (float(s.detectorColCount) * float(s.detectorColSize)
                   * float(s.sid) / float(s.sdd))
        half = fan_fov / 2.0
        cov = (1.0 if rp <= half else float(2.0 * np.arcsin(half / rp) / np.pi))
        if cov < 0.99:
            warn.append(
                f"周边采样点 r={rp:.0f}mm 超出扇束半宽 {half:.0f}mm（等中心 FOV {fan_fov:.0f}mm）："
                f"仅 {cov * 100:.0f}% 的角度被照射，CTDI_periph/CTDI_w 偏低；"
                "建议改小 phantom_diameter_mm（如 160mm 头模）或扩大平板/SID")
    except Exception:                                 # noqa: BLE001
        pass

    per100 = {}
    if mAs_per_rotation:
        f = 100.0 / float(mAs_per_rotation)
        per100 = {"center": mean100["center"] * f, "periph": mean100["periph"] * f,
                  "ctdi_w_mean": ctdi_w_mean * f, "ctdi_w": ctdi_w * f}

    out = {
        "phantom": {"diameter_mm": diam, "length_mm": ns * thick,
                    "material_hu": float(c.material_hu), "n_voxel": n,
                    "slice_thickness_mm": thick},
        "integration_mm": float(c.integration_mm),
        "periph_radius_mm": rp,
        "fan_fov_mm": (None if not np.isfinite(fan_fov) else fan_fov),
        "periph_fan_coverage": (None if not np.isfinite(cov) else cov),
        "warnings": warn,
        "n_views_360": int(n_views_eff),
        "n_views_used": int(n_used),
        "view_density_per_360": float(c.view_density_per_360),
        "mAs_per_rotation": (None if mAs_per_rotation is None else float(mAs_per_rotation)),
        "beam_width_mm": (None if not np.isfinite(nT) else nT),
        "mean100": mean100,
        "ctdi100": ctdi100,
        "ctdi_w_mean100": float(ctdi_w_mean),
        "ctdi_w": float(ctdi_w),
        "per_100mAs": per100,
        "unit": "mGy",
        "definitions": {
            "mean100": "中央 100 mm 层厚内的平均吸收剂量（center: r≈0；periph: r=periph_radius 的 4 点平均）",
            "ctdi100": "严格 CTDI100 = mean100 × 100mm / nT（nT=本几何在等中心的束宽）",
            "ctdi_w": "1/3·center + 2/3·periph（对 ctdi100）",
            "ctdi_w_mean100": "同上加权，但用 mean100（宽束等价，平板 C 臂更直观）",
            "which_to_quote": (
                "nT ≤ 100mm 时 ctdi_w 与临床 CTDIw 定义同形，可直接对比；"
                "本平板 C 臂 nT 往往 > 150mm（远宽于 100mm 电离室），此时 ctdi100 因除 nT 而系统性偏小，"
                "应引用 ctdi_w_mean100（≈ 宽束 CTDIw）"),
            "caveats": "本引擎无散射；CTDI 标准为电离室+PMMA 体模+bowtie；仅可同口径比较",
        },
    }
    if return_dose:
        out["dose"] = dose
        out["dose_grid"] = {"shape": list(dose.shape), "voxel_mm": [vxy, vxy, thick],
                            "z_slab_index": [int(z0), int(z0 + k)]}
    return out


# --------------------------------------------------------------------------- 空气比释动能 / DAP
#: 空气质能吸收系数 (μ_en/ρ)_air [cm²/g]（NIST，10~200 keV）
AIR_MU_EN_RHO_KEV = np.array([
    [10.0, 4.742], [15.0, 1.289], [20.0, 0.5501], [30.0, 0.1537],
    [40.0, 0.06839], [50.0, 0.04098], [60.0, 0.02797], [80.0, 0.02404],
    [100.0, 0.02325], [150.0, 0.02514], [200.0, 0.02623],
])

#: 1 keV = 1.602176634e-16 J
_KEV_TO_J = 1.602176634e-16


def air_mu_en_rho(kev):
    """(μ_en/ρ)_air [cm²/g]，对数-对数插值（区间外取端点）。"""
    k = np.clip(np.asarray(kev, dtype=float),
                AIR_MU_EN_RHO_KEV[0, 0], AIR_MU_EN_RHO_KEV[-1, 0])
    return np.exp(np.interp(np.log(k), np.log(AIR_MU_EN_RHO_KEV[:, 0]),
                            np.log(AIR_MU_EN_RHO_KEV[:, 1])))


def estimate_air_kerma(
    evec_kev,
    net_ivec,
    *,
    sid_mm: float,
    sdd_mm: float,
    n_col: int,
    n_row: int,
    col_size_mm: float,
    row_size_mm: float,
    ref_dist_mm: float | None = None,
    n_frames: int = 1,
    mAs_total: float | None = None,
    cell: str = "center",
):
    """由滤过后的**每 cell 注量谱**求 **参考点空气比释动能 Ka,r** 与 **DAP**。

    `net_ivec` 是 CatSim `cfg.spec.netIvec`：shape `(n_cells, n_Ebin)`，
    单位 **photons/view/mm² at 1 m**（已含 mA×viewTime、bowtie 与平滤过）。
    `evec_kev` 为对应能量 bin 中心 [keV]。

    定义与换算：
      ``K_air = Σ_E Φ_E · E · (μ_en/ρ)_air``  （Φ 从 1 m 平方反比搬到目标距离）
      ``Ka,r`` 取 **参考点** = 等中心向源侧 15 mm（IEC 60601-2-43）→ r = SID − 15 mm
      ``DAP = K_air(等中心面) × 该处照射野面积``  →  mGy·cm²

    `cell="center"` 用中心探测单元（对应过等中心的那条射线）；
    `cell="mean"` 用所有单元平均（对整个扇束取平均）。
    """
    e = np.asarray(evec_kev, dtype=float).ravel()
    net = np.asarray(net_ivec, dtype=float)
    net = net.reshape(-1, e.size) if net.ndim >= 2 else net[None, :]
    if cell == "center":
        phi = net[net.shape[0] // 2]
    elif cell == "mean":
        phi = net.mean(axis=0)
    else:
        raise ValueError(f"cell 须为 'center'/'mean'，收到 {cell!r}")

    if ref_dist_mm is None:
        ref_dist_mm = float(sid_mm) - 15.0

    def _kerma_mGy(phi_per_mm2: np.ndarray, dist_mm: float) -> float:
        # Φ[1/mm²] → [1/m²]；E[J]；(μ_en/ρ)[cm²/g] → [m²/kg]（×0.1）→ Gy → mGy
        gy = float(np.sum(phi_per_mm2 * 1e6 * e * _KEV_TO_J * air_mu_en_rho(e) * 0.1))
        return gy * (1000.0 / float(dist_mm)) ** 2 * 1e3

    k_ref = _kerma_mGy(phi, ref_dist_mm)
    k_iso = _kerma_mGy(phi, sid_mm)
    field_w = float(n_col) * float(col_size_mm) * float(sid_mm) / float(sdd_mm)
    field_h = float(n_row) * float(row_size_mm) * float(sid_mm) / float(sdd_mm)
    dap = k_iso * (field_w / 10.0) * (field_h / 10.0)          # mGy·cm²

    out = {
        "k_ref_per_view": k_ref,
        "k_iso_per_view": k_iso,
        "dap_per_view": dap,
        "k_ref_total": k_ref * n_frames,
        "k_iso_total": k_iso * n_frames,
        "dap_total": dap * n_frames,
        "n_frames": int(n_frames),
        "ref_dist_mm": float(ref_dist_mm),
        "field_mm": (field_w, field_h),
        "field_cm2": (field_w / 10.0) * (field_h / 10.0),
        "mean_kev": float(np.sum(phi * e) / max(np.sum(phi), 1e-30)),
        "cell": cell,
        "unit": "mGy",
        "dap_unit": "mGy*cm2",
        "definitions": {
            "k_ref": "参考点(=等中心向源侧 15mm)自由空气比释动能 Ka,r，累积本协议全部帧",
            "dap": "剂量面积乘积 = Ka(等中心面) × 该面照射野面积，mGy·cm²（= mGy·cm²/1000 Gy·cm²）",
            "note": "自由空气量，无体模/无散射；与设备剂量报告同为『空气比释动能』口径",
        },
    }
    if mAs_total:
        out["per_100mAs"] = {
            "k_ref": k_ref * n_frames * 100.0 / float(mAs_total),
            "dap": dap * n_frames * 100.0 / float(mAs_total),
        }
        out["mAs_total"] = float(mAs_total)
    return out
