"""XCIST (gecatsim) 后端整体测试：数值一致性 + 旋转可视化产物。

分两部分：
1. `test_xcist_drr_mono_matches_torch` —— mono 模式的线积分必须与 `TorchDRR`
   在**斜视角**下一致（NCC > 0.98）。这是几何标定的回归闸门：
   体模的 X/Y 轴互换、以及"体素第 2 轴 = 世界 -Y"这两个坑在 alpha = 0°/180°
   下都**不可见**，只有斜视角能暴露（见 `phantom.FLIP_Y_IN_VP`）。
2. `test_integ_xcist` —— 仿照 `test_integ`，用 XcistDRR 跑完整 `RotateDSA`
   流程并落盘可视化旋转图像（tif / gif / 每帧 png / label / depth / json）。

说明
- gecatsim 需要在本地编译 C 库（无 py3.14 wheel），未安装时整体 skip，
  避免 CI 因缺依赖而失败。
- 分辨率/帧数刻意取小（128²、6 帧）：gecatsim 是 CPU 光线追踪，
  200³ 体模 + 128² 平板约 1.5~4 s/view。
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("gecatsim", reason="gecatsim (XCIST) 未安装，跳过 xcist 后端测试")

from generate_4d_heart.rotate_dsa import RotateDSA
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.rotate_dsa.rotate_drr import (
    CArmGeometry,
    RotatedParameters,
    TorchDRR,
    XcistDRR,
)
from generate_4d_heart.xcist.dose import (
    CtdiConfig, DoseConfig, estimate_air_kerma, estimate_ctdi_like,
)
from generate_4d_heart.xcist.geometry import XcistGeometry
from generate_4d_heart.xcist.phantom import _sample_to_canonical, iso_from_label
from generate_4d_heart.xcist.sim import XcistPhysics, XcistSim

from utils import ReaderName, SimulatorName, reader_factory

#: 测试用小型平板几何：128² @ 1.0mm（等中心 FOV ≈ 65mm），兼顾速度与解剖覆盖。
XCIST_GEOM = CArmGeometry(
    sdd=1200.0,
    sod=760.0,
    height=128,
    width=128,
    delx=1.0,
    dely=1.0,
    orientation_type="AP",
)

#: 帧数/角度：6 帧、步进 3°（-30° → -15°）。全部为斜视角。
XCIST_FRAMES = 6
XCIST_ALPHA_START = -30.0
XCIST_ANGULAR_VELOCITY = 30.0

#: 剂量测试用的极小网格/能量分箱（仅验证管路与产物，非精度）
DOSE_CFG = DoseConfig(n_voxel=32, fov_mm=256.0, slice_thickness_mm=8.0)
DOSE_SLICES = 8
DOSE_SUBSAMPLE = 3
DOSE_ENERGY_BINS = 2

# ── 真实采集协议：90kV / 50mA / 0.1mm Cu / 3.3ms 每帧；60 fps、75°/s、3 s ──
REAL_FPS = 60.0
REAL_FRAMES = 180            # 60 fps × 3 s
REAL_ALPHA_START = -112.5    # 225° 扫角大致居中于 0°
REAL_ALPHA_END = 112.5
REAL_ANGULAR_STEP = 1.25     # 75°/s ÷ 60 fps
REAL_MAS_TOTAL = 50.0 * 3.3e-3 * REAL_FRAMES   # ≈ 29.7 mAs

#: 真实协议剂量网格：32³×16 层（8mm）——粗网格换速度，以跑**全 180 帧 view 基线**
REAL_DOSE_CFG = DoseConfig(n_voxel=32, fov_mm=256.0, slice_thickness_mm=8.0)
REAL_DOSE_SLICES = 16
REAL_DOSE_ENERGY_BINS = 4


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    b = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


@pytest.fixture(scope="module")
def xcist_rotate_cfg() -> RotatedParameters:
    return RotatedParameters(
        total_frame=XCIST_FRAMES,
        fps=10.0,
        angular_velocity=XCIST_ANGULAR_VELOCITY,
        alpha_start=XCIST_ALPHA_START,
    )


@pytest.fixture(scope="module")
def xcist_reader():
    """RBFReader（体积+标签，含造影剂），模块内复用一次以减少加载时间。"""
    _, reader = reader_factory(
        SimulatorName.STATIC_IODINE_CONTRAST,
        ReaderName.RBF_READER,
        cororonary_need_load="LCA",
    )
    yield reader
    from generate_4d_heart.rotate_dsa.data_reader.data_reader import cleanup_gpu_memory

    cleanup_gpu_memory()


@pytest.fixture(scope="module")
def xcist_drr(xcist_rotate_cfg: RotatedParameters) -> XcistDRR:
    return XcistDRR(
        c_arm_cfg=XCIST_GEOM,
        rotate_cfg=xcist_rotate_cfg,
        mode="mono",
        monochromatic_kev=70.0,
    )


def test_xcist_drr_mono_matches_torch(
    xcist_reader,
    xcist_drr: XcistDRR,
    xcist_rotate_cfg: RotatedParameters,
):
    """mono 模式 vs TorchDRR：斜视角下逐帧线积分一致（几何标定回归闸门）。"""
    read_res = xcist_reader.get_dynamic_data(CardiacPhase.from_time(0.0, 1.0), "LCA")
    volume = read_res.coronary.volume_centering
    coronary = read_res.coronary.label_centering

    torch_drr = TorchDRR(c_arm_cfg=XCIST_GEOM, rotate_cfg=xcist_rotate_cfg, patch_size=128)

    for frame in (0, XCIST_FRAMES // 2):
        alpha, _, _ = xcist_rotate_cfg.get_rotation_angle_at_frame(frame)
        # 前提：必须是斜视角，否则深度轴镜像类 bug 不可见
        assert abs(alpha) > 1.0, f"该测试要求 alpha != 0，实际 {alpha}"

        e_xcist = xcist_drr.get_projection_at_frame(frame, volume, coronary)
        e_torch = torch_drr.get_projection_at_frame(frame, volume, coronary)
        assert e_xcist.shape == e_torch.shape == (1, 1, XCIST_GEOM.height, XCIST_GEOM.width)

        x = e_xcist.squeeze().cpu().numpy()
        t = e_torch.squeeze().cpu().numpy()
        assert np.isfinite(x).all() and np.isfinite(t).all()

        ncc = _ncc(x, t)
        assert ncc > 0.98, f"frame {frame} (alpha={alpha:.2f}°): NCC={ncc:.4f} 过低"

        # 幅值（线积分尺度）也应接近：mono 下 xcist 精确复现 ∫μds
        rel = abs(float(x.mean()) - float(t.mean())) / max(float(t.mean()), 1e-6)
        assert rel < 0.05, f"frame {frame}: 平均线积分相对误差 {rel:.4f} 过大"

        print(f"[xcist] frame {frame} alpha={alpha:+.2f}° NCC={ncc:.4f} "
              f"mean E: xcist={x.mean():.4f} torch={t.mean():.4f}")


def test_integ_xcist(
    output_root_dir: Path,
    xcist_reader,
    xcist_drr: XcistDRR,
):
    """完整 RotateDSA 流程（XcistDRR mono）→ 可视化旋转图像 + 几何 json。"""
    dsa = RotateDSA(reader=xcist_reader, drr=xcist_drr)

    output_dir = output_root_dir / "xcist" / "rbf_reader_static_iodine_contrast_LCA"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output = dsa.run_and_save(
        output_dir=output_dir,
        coronary_type="LCA",
        gif_fps=5,
    )

    # ── 几何 json（与 test_integ 相同约定） ──
    geometry_json = dsa.get_geometry_json("LCA")
    assert "frames" in geometry_json
    assert "c_arm_geometry" in geometry_json
    assert "rotate_parameters" in geometry_json
    assert len(geometry_json["frames"]) == xcist_drr.rotate_cfg.total_frame
    assert geometry_json["additional_config"]["renderer"] == "xcist"
    assert geometry_json["additional_config"]["mode"] == "mono"

    for frame_info in geometry_json["frames"]:
        assert isinstance(frame_info["alpha_degree"], float)
        assert isinstance(frame_info["beta_degree"], float)
        assert isinstance(frame_info["phase"], (int, float))

    # ── 数值产物 ──
    T = xcist_drr.rotate_cfg.total_frame
    assert output.raw_frames.shape == (T, XCIST_GEOM.height, XCIST_GEOM.width)
    assert output.vis_frames.shape == output.raw_frames.shape
    assert np.isfinite(output.raw_frames).all()
    # Beer-Lambert 后强度必须落在 (0, 1]
    assert output.raw_frames.min() >= 0.0
    assert output.raw_frames.max() <= 1.0 + 1e-6
    # 标签/深度：至少有内容且有限
    assert (output.labels > 0).any(), "labels 全为 0"
    assert np.isfinite(output.depth_maps).any(), "depth_maps 全为 nan"
    # 旋转确实发生（首末帧应有差异）
    assert float(np.abs(output.raw_frames[0] - output.raw_frames[-1]).mean()) > 1e-4

    # ── 落盘产物 ──
    assert (output_dir / "rotate_dsa_raw.tif").exists()
    assert (output_dir / "rotate_dsa_raw.gif").exists()
    assert (output_dir / "rotate_dsa.gif").exists()
    assert (output_dir / "label.gif").exists()
    assert (output_dir / "depth_map.gif").exists()
    assert (output_dir / "rotate_dsa.json").exists()
    assert (output_dir / "label.npz").exists()
    assert (output_dir / "depth_map.npz").exists()

    saved = json.loads((output_dir / "rotate_dsa.json").read_text())
    assert saved["additional_config"]["renderer"] == "xcist"
    assert len(saved["frames"]) == T

    print(f"[xcist] RotateDSA 产物已写出: {output_dir}")
    print(f"[xcist] raw range [{output.raw_frames.min():.4f}, {output.raw_frames.max():.4f}], "
          f"labels 非零 {int((output.labels > 0).sum())} px")


def test_xcist_dose_disabled_by_default(xcist_drr: XcistDRR):
    """默认不启用剂量：无配置、无副作用（不引入额外耗时/文件）。"""
    assert xcist_drr.dose_config_dict() is None
    assert xcist_drr.estimate_run_dose() is None
    assert xcist_drr.get_additional_config()["dose"] is None


@pytest.fixture(scope="module")
def xcist_drr_dose(xcist_rotate_cfg: RotatedParameters) -> Iterator[XcistDRR]:
    """启用剂量估计的 XcistDRR（小网格/低分箱，仅验证管路）。"""
    drr = XcistDRR(
        c_arm_cfg=XCIST_GEOM,
        rotate_cfg=xcist_rotate_cfg,
        mode="mono",
        dose_cfg=DOSE_CFG,
        dose_slices=DOSE_SLICES,
        dose_subsample=DOSE_SUBSAMPLE,
        dose_energy_bins=DOSE_ENERGY_BINS,
    )
    yield drr
    drr._dose_sim = None


def test_integ_xcist_dose(
    output_root_dir: Path,
    xcist_reader,
    xcist_drr_dose: XcistDRR,
):
    """启用剂量后：run_and_save 同步输出 dose.npy / dose.mGy + json 元数据。"""
    dsa = RotateDSA(reader=xcist_reader, drr=xcist_drr_dose)
    output_dir = output_root_dir / "xcist" / "dose_LCA"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    dsa.run_and_save(output_dir=output_dir, coronary_type="LCA", gif_fps=5)

    assert (output_dir / "dose.npy").exists()
    assert (output_dir / "dose.mGy").exists()
    dose = np.load(output_dir / "dose.npy")
    assert dose.shape == (DOSE_CFG.n_voxel, DOSE_CFG.n_voxel, DOSE_SLICES)
    assert np.isfinite(dose).all()
    assert dose.max() > 0.0, "剂量全为 0"
    # .mGy 应为 raw float32（gecatsim 官方写法的等价物）
    assert (output_dir / "dose.mGy").stat().st_size == dose.size * 4

    saved = json.loads((output_dir / "rotate_dsa.json").read_text())
    assert "dose" in saved
    dmeta = saved["dose"]
    # 未提供参考 CT → 用 DRR 的 μ 体积近似
    assert dmeta["hu_source"] == "mu_approx"
    assert dmeta["n_views_total"] == XCIST_FRAMES
    assert dmeta["n_views_sampled"] == DOSE_SUBSAMPLE
    assert dmeta["mAs_total"] > 0
    assert sorted(dmeta["stats"]) == ["iso", "whole_max", "whole_mean", "whole_p99_9"]
    for k, v in dmeta["stats"].items():
        assert np.isfinite(v), f"dose stats[{k}] 非有限"
    assert dmeta["stats"]["whole_mean"] > 0
    assert saved["additional_config"]["dose"]["enabled"] is True
    # 同一份剂量报告里还带 CTDI 口径（360° 标准化体模）
    assert "ctdi" in dmeta and "ctdi_w" in dmeta["ctdi"]
    assert np.isfinite(dmeta["ctdi"]["ctdi_w"]) and dmeta["ctdi"]["ctdi_w"] > 0
    assert saved["additional_config"]["dose"]["ctdi"] is not None
    print(f"[xcist] dose(mu_approx): mean={dmeta['stats']['whole_mean']:.4g} "
          f"p99.9={dmeta['stats']['whole_p99_9']:.4g} mGy, mAs={dmeta['mAs_total']:.2f}, "
          f"CTDIw={dmeta['ctdi']['ctdi_w']:.4g} mGy")


def test_xcist_dose_with_reference_ct(xcist_drr_dose: XcistDRR):
    """提供参考 CT(HU) 时优先用它，并支持 ROI 统计。"""
    n, ns = DOSE_CFG.n_voxel, DOSE_SLICES
    hu = np.zeros((n, n, ns), dtype=np.float32)
    hu[:8, n // 2 - 4:n // 2 + 4, :] = 1000.0        # 内切圆内的骨块
    xcist_drr_dose.set_dose_reference_hu(hu)

    roi = np.zeros((n, n, ns), dtype=bool)
    roi[n // 2 - 4:n // 2 + 4, n // 2 - 4:n // 2 + 4, :] = True
    res = xcist_drr_dose.estimate_run_dose(roi_mask=roi)

    assert res is not None
    assert res["meta"]["hu_source"] == "reference_ct"
    assert res["dose"].shape == (n, n, ns)
    assert np.isfinite(res["dose"]).all()
    assert res["dose"].max() > 0
    assert "roi_mean" in res["meta"]["stats"]
    assert np.isfinite(res["meta"]["stats"]["roi_mean"])
    print(f"[xcist] dose(reference_ct): whole_mean={res['meta']['stats']['whole_mean']:.4g}, "
          f"roi_mean={res['meta']['stats']['roi_mean']:.4g} mGy")

    # 复位，避免影响其它用例
    xcist_drr_dose._dose_hu_ref = None
    xcist_drr_dose._dose_hu_fallback = None


def _real_protocol_cfg() -> RotatedParameters:
    """真实协议：180 帧、-112.5° → +112.5°（步进 1.25°、角速度 75°/s、fps 60）。"""
    return RotatedParameters.from_alpha_range(
        total_frame=REAL_FRAMES,
        alpha_start=REAL_ALPHA_START,
        alpha_end=REAL_ALPHA_END,
        fps=REAL_FPS,
        include_end=False,      # 180 帧 × 1.25° = 225°（不重复末帧）
    )


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读 NIfTI，返回 (data, affine)。"""
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)          # type: ignore[attr-defined]
    affine = np.asarray(img.affine, dtype=float)   # type: ignore[attr-defined]
    return data, affine


def _real_dose_hu(volume_dvf_dir: Path) -> np.ndarray:
    """真实 CTA → 剂量网格上的 HU 体 (N,N,S)（以冠脉中心为 iso，粗网格）。"""
    hu, aff = _load_nifti(volume_dvf_dir / "image.nii.gz")
    cor, _ = _load_nifti(volume_dvf_dir / "coronary.nii.gz")
    iso = iso_from_label(cor, aff)

    n, ns = REAL_DOSE_CFG.n_voxel, REAL_DOSE_SLICES
    vxy = REAL_DOSE_CFG.fov_mm / n
    vz = REAL_DOSE_CFG.slice_thickness_mm
    half = (REAL_DOSE_CFG.fov_mm / 2.0, REAL_DOSE_CFG.fov_mm / 2.0, ns * vz / 2.0)
    hu_gzyx = _sample_to_canonical(hu, aff, iso, half, (vxy, vxy, vz), cval=-1000.0)
    hu_gzyx[hu_gzyx < -800] = -1000.0          # 体外/肺 → 空气（HU=0 会被剂量代码当水）
    return np.ascontiguousarray(np.transpose(hu_gzyx, (2, 1, 0)))    # (N,N,S)=(x,y,z)


def test_xcist_dose_real(
    output_root_dir: Path,
    xcist_reader,
    volume_dvf_dir: Path,
):
    """真实协议：90kV / 50mA / 0.1mmCu / 3.3ms；60fps、75°/s、3s（180 帧、225°、29.7mAs）。

    1. 协议自检（帧数/步进/扫角/mAs/能谱/滤过=代码默认 XcistPhysics）；
    2. 用该协议**投影**（抽 4 帧覆盖整个扫角，含斜视角）+ 与 TorchDRR 一致性；
    3. 用真实 CTA 作为剂量沉积介质，跑**全 180 帧 view**的剂量基线并落盘。
    """
    # ── 1) 协议自检 ──
    rcfg = _real_protocol_cfg()
    assert rcfg.total_frame == REAL_FRAMES == 180
    assert rcfg.fps == REAL_FPS == 60.0
    assert np.isclose(rcfg.angular_velocity, 75.0)
    alpha = [rcfg.get_rotation_angle_at_frame(f)[0] for f in (0, 1, REAL_FRAMES - 1)]
    assert np.isclose(alpha[0], REAL_ALPHA_START)
    assert np.isclose(alpha[1] - alpha[0], REAL_ANGULAR_STEP)
    assert np.isclose(alpha[-1] - alpha[0], (REAL_FRAMES - 1) * REAL_ANGULAR_STEP)

    physics = XcistPhysics()          # 默认即真实协议
    assert physics.mA == 50.0
    assert np.isclose(physics.exposure_s, 3.3e-3)
    assert physics.fps == 60.0
    assert physics.flat_filter_mm == ["cu", 0.1]
    assert physics.spectrum_filename == "tungsten_tar7.0_90_filt.dat"
    assert np.isclose(physics.mA * physics.exposure_s * REAL_FRAMES, REAL_MAS_TOTAL, rtol=1e-6)
    print(f"[xcist-real] 协议: {REAL_FRAMES} 帧 × {REAL_ANGULAR_STEP}° = "
          f"{REAL_FRAMES * REAL_ANGULAR_STEP:.1f}°, 总 {REAL_MAS_TOTAL:.2f} mAs")

    # ── 2) 真实协议投影（覆盖全扫角）+ 与 TorchDRR 一致性 ──
    xdrr = XcistDRR(
        c_arm_cfg=XCIST_GEOM,
        rotate_cfg=rcfg,
        mode="mono",
        physics=physics,
        dose_cfg=REAL_DOSE_CFG,
        dose_slices=REAL_DOSE_SLICES,
        dose_subsample=REAL_FRAMES,      # =1：全协议逐帧（不缩放）
        dose_energy_bins=REAL_DOSE_ENERGY_BINS,
    )
    read_res = xcist_reader.get_dynamic_data(CardiacPhase.from_time(0.0, 1.0), "LCA")
    volume = read_res.coronary.volume_centering
    coronary = read_res.coronary.label_centering

    probe_frames = (0, 60, 120, REAL_FRAMES - 1)
    for f in probe_frames:
        e = xdrr.get_projection_at_frame(f, volume, coronary)
        arr = e.squeeze().cpu().numpy()
        assert e.shape == (1, 1, XCIST_GEOM.height, XCIST_GEOM.width)
        assert np.isfinite(arr).all() and arr.min() >= 0.0
        assert arr.max() > 0.0
        print(f"[xcist-real] frame {f:3d}: alpha={alpha[0] + f * REAL_ANGULAR_STEP:+7.2f}° "
              f"E∈[{arr.min():.3f}, {arr.max():.3f}] mean={arr.mean():.3f}")

    f_chk = probe_frames[1]                     # alpha ≈ -37.5°，斜视角
    torch_drr = TorchDRR(c_arm_cfg=XCIST_GEOM, rotate_cfg=rcfg, patch_size=128)
    e_x = xdrr.get_projection_at_frame(f_chk, volume, coronary).squeeze().cpu().numpy()
    e_t = torch_drr.get_projection_at_frame(f_chk, volume, coronary).squeeze().cpu().numpy()
    ncc = _ncc(e_x, e_t)
    assert ncc > 0.98, f"frame {f_chk}: NCC={ncc:.4f} 过低"
    print(f"[xcist-real] vs TorchDRR @frame {f_chk}: NCC={ncc:.4f}")

    # ── 3) 全协议剂量（180 view，真实 CTA 作为沉积介质） ──
    n, ns = REAL_DOSE_CFG.n_voxel, REAL_DOSE_SLICES
    hu_xyz = _real_dose_hu(volume_dvf_dir)
    assert hu_xyz.shape == (n, n, ns)
    xdrr.set_dose_reference_hu(hu_xyz)

    res = xdrr.estimate_run_dose()
    assert res is not None
    meta, dose = res["meta"], res["dose"]
    assert meta["hu_source"] == "reference_ct"
    assert meta["n_views_total"] == REAL_FRAMES
    assert meta["n_views_sampled"] == REAL_FRAMES
    assert np.isclose(meta["scale_factor"], 1.0)
    assert np.isclose(meta["mAs_total"], REAL_MAS_TOTAL)
    assert meta["energy_bins"] == REAL_DOSE_ENERGY_BINS
    assert meta["spectrum_filename"] == "tungsten_tar7.0_90_filt.dat"
    assert meta["flat_filter_mm"] == ["cu", 0.1]
    assert dose.shape == (n, n, ns)
    assert np.isfinite(dose).all() and dose.max() > 0.0
    for k, v in meta["stats"].items():
        assert np.isfinite(v), f"dose stats[{k}] 非有限"
    assert 1e-3 < meta["stats"]["whole_mean"] < 10.0, meta["stats"]

    out_dir = output_root_dir / "xcist" / "dose_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "dose.npy", dose)
    dose.tofile(out_dir / "dose.mGy")
    (out_dir / "dose_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[xcist-real] 剂量(全 {REAL_FRAMES} view, {REAL_MAS_TOTAL:.2f} mAs): "
          f"mean={meta['stats']['whole_mean']:.4g} "
          f"p99.9={meta['stats']['whole_p99_9']:.4g} "
          f"max={meta['stats']['whole_max']:.4g} iso={meta['stats']['iso']:.4g} mGy "
          f"-> {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# pig_data_drr 整流程报告：旋转图像 + 患者/心脏 ROI 剂量 + CTDI 口径
# ──────────────────────────────────────────────────────────────────────────────
#: 剂量网格：48²×48 层（5mm）覆盖 320mm FOV（够装下整个胸腔）
PIG_DOSE_CFG = DoseConfig(n_voxel=48, fov_mm=320.0, slice_thickness_mm=5.0)
PIG_DOSE_SLICES = 48
PIG_DOSE_SUBSAMPLE = 12          # 60 帧里均匀取 12 个视角沉积再线性缩放
PIG_DOSE_ENERGY_BINS = 4


def _grid_sample(
    data,
    affine,
    iso_world,
    n: int,
    ns: int,
    vxy: float,
    vz: float,
    *,
    order: int = 1,
    cval: float = -1000.0,
) -> np.ndarray:
    """任意 (index-order) 体积 → 剂量网格 `(N, N, S) = (x, y, z)`。"""
    half = (vxy * n / 2.0, vxy * n / 2.0, vz * ns / 2.0)
    g = _sample_to_canonical(
        np.asarray(data), np.asarray(affine, dtype=float),
        np.asarray(iso_world, dtype=float), half, (vxy, vxy, vz),
        order=order, cval=cval,
    )
    return np.ascontiguousarray(np.transpose(g, (2, 1, 0)))


def _save_dose_figure(path: Path, dose: np.ndarray, roi: np.ndarray | None = None) -> None:
    """患者剂量报告图：三向 MIP + 中间层（带心腔 ROI 轮廓）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.asarray(dose, dtype=float)
    panels = [
        ("axial MIP (z)", d.max(axis=2).T, None),
        ("coronal MIP (y)", d.max(axis=1).T, None),
        ("sagittal MIP (x)", d.max(axis=0).T, None),
    ]
    vmax = float(np.percentile(d[d > 0], 99)) if (d > 0).any() else 1.0
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (title, img, _) in zip(axes.ravel(), panels):
        im = ax.imshow(img, cmap="inferno", vmin=0.0, vmax=vmax, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.9)

    ax = axes.ravel()[3]
    mid = d[:, :, d.shape[2] // 2].T
    im = ax.imshow(mid, cmap="inferno", vmin=0.0, vmax=vmax, origin="lower")
    if roi is not None and roi.any():
        r = np.asarray(roi)[:, :, np.asarray(roi).shape[2] // 2].T
        ax.contour(r, levels=[0.5], colors="cyan", linewidths=1.0)
        ax.set_title(f"mid axial slice + heart ROI\nmean={d[roi].mean():.3g} / "
                     f"p99.9={np.percentile(d[roi], 99.9):.3g} mGy")
    else:
        ax.set_title("mid axial slice")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.9)
    fig.suptitle("XCIST absorbed dose (mGy, whole rotation)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _small_geom(n: int = 96, pix: float = 2.0) -> XcistGeometry:
    """小面板几何（面板尺寸成比例，通量/束宽含义不变）→ 快速口径测试用。"""
    return XcistGeometry(sid=760.0, sdd=1200.0, n_col=n, n_row=n,
                         col_size=pix, row_size=pix, orientation="AP")


def test_xcist_ctdi_slice_thickness_invariance():
    """CTDI-like 必须与**层厚无关**（真实体素体积归一化的回归闸门）。

    历史 bug：质量用各向同性 `(fov/n)^3`（官方 img2vol 的隐含假设），而能量沉积
    用 `(vxy, vz)` → 剂量正比于层厚（2/4/5/8 mm 实测比例 1:2:2.5:4）。
    修复后同一体模在 2 mm 与 8 mm 下应一致。
    """
    geom = _small_geom(96, 2.0)              # 192mm 面板
    ref = None
    for thick in (2.0, 8.0):
        sim = XcistSim(geometry=geom, physics=XcistPhysics(
            flat_filter_mm=["cu", 0.1], mA=50.0, exposure_s=3.3e-3,
            fps=60.0, energy_count=2))
        cfg = CtdiConfig(n_voxel=32, slice_thickness_mm=thick,
                         phantom_diameter_mm=160.0, detector_downsample=1,
                         min_views=24, max_views=24, integration_mm=40.0)
        cfg.view_density_per_360 = 187 / 200.0
        res = estimate_ctdi_like(sim._ct.cfg, cfg,
                                 mAs_per_rotation=50 * 3.3e-3 * cfg.n_views())
        val = res["per_100mAs"]["ctdi_w_mean"]
        assert np.isfinite(val) and val > 0
        print(f"[xcist-ctdi] thick={thick}mm -> per100mAs={val:.3f} mGy")
        if ref is None:
            ref = val
        else:
            assert abs(val / ref - 1.0) < 0.10, (
                f"CTDI 随层厚变化（{thick}mm: {val:.3f} vs {ref:.3f}）→ 质量归一化漏乘层厚")


def test_xcist_air_kerma_and_dap():
    """Ka,r / DAP：线性性、Cu 过滤单调性、DAP=Ka(iso)×野面积，并与设备报告量级对照。"""
    geom = _small_geom(96, 2.0)
    sim = XcistSim(geometry=geom, physics=XcistPhysics(
        flat_filter_mm=["cu", 0.1], mA=50.0, exposure_s=3.3e-3,
        fps=60.0, energy_count=20))
    e, net = sim.spectrum()
    kw = dict(sid_mm=760.0, sdd_mm=1200.0, n_col=96, n_row=96,
              col_size_mm=2.0, row_size_mm=2.0)

    r1 = estimate_air_kerma(e, net, n_frames=100, mAs_total=50 * 3.3e-3 * 100, **kw)
    r2 = estimate_air_kerma(e, net, n_frames=200, mAs_total=50 * 3.3e-3 * 200, **kw)
    assert np.isclose(r2["k_ref_total"], 2.0 * r1["k_ref_total"], rtol=1e-6)
    assert np.isclose(r2["dap_total"], 2.0 * r1["dap_total"], rtol=1e-6)
    # DAP = Ka(iso) × 野面积(cm²)
    assert np.isclose(r1["dap_per_view"],
                      r1["k_iso_per_view"] * r1["field_cm2"], rtol=1e-6)
    # 参考点比等中心更靠近源 → 剂量更大
    assert r1["k_ref_per_view"] > r1["k_iso_per_view"]
    # 每 100 mAs 归一
    assert np.isclose(r1["per_100mAs"]["k_ref"],
                      r1["k_ref_total"] * 100.0 / r1["mAs_total"], rtol=1e-6)

    # Cu 过滤越厚剂量越低（单调）
    vals = []
    for cu in (0.0, 0.1, 0.2):
        s = XcistSim(geometry=geom, physics=XcistPhysics(
            flat_filter_mm=([] if cu == 0 else ["cu", cu]), mA=50.0,
            exposure_s=3.3e-3, fps=60.0, energy_count=20))
        ee, nn = s.spectrum()
        vals.append(estimate_air_kerma(ee, nn, n_frames=187,
                                       mAs_total=50 * 3.3e-3 * 187, **kw)["k_ref_total"])
    assert vals[0] > vals[1] > vals[2], f"Cu 过滤单调性异常: {vals}"

    print(f"[xcist-kerma] 187 帧 50mA 0.1Cu: Ka,r={vals[1]:.3f} mGy, "
          f"DAP={r1['dap_total'] / 1000:.3f} Gy·cm², 均值能量={r1['mean_kev']:.1f} keV")
    print(f"[xcist-kerma] Cu 0/0.1/0.2 的 Ka,r = {[round(v, 3) for v in vals]} mGy")


def _save_ctdi_figure(path: Path, ctdi: dict) -> None:
    """CTDI 体模图：中间层剂量分布 + 径向剖面（中心/周边测量点）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dose = np.asarray(ctdi["dose"], dtype=float)
    nz = dose.shape[2] // 2
    axial = dose[:, :, nz]
    n = axial.shape[0]
    vxy = float(ctdi["phantom"]["diameter_mm"]) / n
    xv = (np.arange(n) - (n - 1) / 2.0) * vxy
    rp = float(ctdi["periph_radius_mm"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    im = axes[0].imshow(axial.T, cmap="inferno", origin="lower",
                        extent=[xv[0], xv[-1], xv[0], xv[-1]])
    axes[0].plot(0, 0, "c+", ms=14, mew=2, label="center")
    for ang in (0, 90, 180, 270):
        axes[0].plot(rp * np.cos(np.deg2rad(ang)), rp * np.sin(np.deg2rad(ang)),
                     "wx", ms=8, mew=2, label=("periph" if ang == 0 else None))
    axes[0].set_title(f"CTDI phantom ø{ctdi['phantom']['diameter_mm']:.0f}mm, "
                      f"{ctdi['n_views_360']} views/360°")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    rr = np.hypot(*np.meshgrid(xv, xv, indexing="ij"))
    bins = np.linspace(0, float(ctdi["phantom"]["diameter_mm"]) / 2.0, 25)
    prof = [axial[(rr >= a) & (rr < b)].mean() if ((rr >= a) & (rr < b)).any() else np.nan
            for a, b in zip(bins[:-1], bins[1:])]
    axes[1].plot(0.5 * (bins[:-1] + bins[1:]), prof, "o-", ms=3)
    axes[1].axvline(rp, color="w", ls="--", lw=1)
    axes[1].set_xlabel("radius (mm)")
    axes[1].set_ylabel("dose at mid-slice (mGy)")
    axes[1].set_title(f"center={ctdi['mean100']['center']:.3g}, "
                      f"periph={ctdi['mean100']['periph']:.3g}, "
                      f"CTDIw={ctdi['ctdi_w']:.3g} mGy")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def test_xcist_pig_report(
    output_root_dir: Path,
    xcist_reader,
    pig_data_drr: TorchDRR,
    volume_dvf_dir: Path,
):
    """用 **pig_data_drr 的真实几何**（474×648 @0.616mm）出完整报告。

    产物（`test/output/xcist/pig_report/`）：

    * `frames/*.png`、`rotate_dsa.gif` —— 旋转图像（选 6 帧覆盖整个扫角）；
    * `dose.npy` / `dose.mGy` / `dose_slices.png` —— 患者吸收剂量（三向 MIP + 心腔 ROI）；
    * `ctdi_phantom.png` —— CTDI 体模中心层分布与径向剖面；
    * `report.json` —— 全部口径：整卷统计、心脏 ROI、CTDI100/CTDIw/每 100mAs。
    """
    out_dir = output_root_dir / "xcist" / "pig_report"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    geom = pig_data_drr.c_arm_cfg
    rcfg = pig_data_drr.rotate_cfg
    assert (geom.height, geom.width) == (474, 648)

    xdrr = XcistDRR(
        c_arm_cfg=geom,
        rotate_cfg=rcfg,
        mode="mono",
        dose_cfg=PIG_DOSE_CFG,
        dose_slices=PIG_DOSE_SLICES,
        dose_subsample=PIG_DOSE_SUBSAMPLE,
        dose_energy_bins=PIG_DOSE_ENERGY_BINS,
    )

    read_res = xcist_reader.get_dynamic_data(CardiacPhase.from_time(0.0, 1.0), "LCA")
    volume = read_res.coronary.volume_centering
    coronary = read_res.coronary.label_centering

    # ── 1) 旋转图像（抽帧覆盖全扫角） ──
    T = rcfg.total_frame
    frames = (0, T // 5, 2 * T // 5, 3 * T // 5, 4 * T // 5, T - 1)
    raw = []
    for f in frames:
        e = xdrr.get_projection_at_frame(f, volume, coronary)
        assert e.shape == (1, 1, geom.height, geom.width)
        arr = e.squeeze().cpu().numpy()
        assert np.isfinite(arr).all() and arr.min() >= 0.0
        raw.append(np.exp(-arr))                     # E=-ln T → T
        alpha = rcfg.get_rotation_angle_at_frame(f)[0]
        print(f"[xcist-pig] frame {f:3d}: alpha={alpha:+7.2f}°  "
              f"T∈[{raw[-1].min():.3f}, {raw[-1].max():.3f}]")
    raw = np.stack(raw)
    assert raw.min() > 0.0 and raw.max() <= 1.0 + 1e-6
    # 旋转确实发生
    assert float(np.abs(raw[0] - raw[-1]).mean()) > 1e-4

    from generate_4d_heart.rotate_dsa.postprocess import postprocess_drr
    from generate_4d_heart.saver import save_gif, save_pngs, save_tif

    vis, _ = postprocess_drr(torch.from_numpy(raw).float(), gamma=0.1)
    save_tif(out_dir / "rotate_dsa_raw.tif", raw)
    save_pngs(out_dir / "frames", vis)
    save_gif(out_dir / "rotate_dsa.gif", vis, fps=5)
    assert (out_dir / "rotate_dsa.gif").exists()
    assert (out_dir / "rotate_dsa_raw.tif").exists()
    assert len(list((out_dir / "frames").glob("*.png"))) == len(frames)

    # 与 torch 后端比一次（斜视角）
    f_chk = T // 2
    torch_drr = TorchDRR(c_arm_cfg=geom, rotate_cfg=rcfg)   # patch_size=256（须 ≤ 面内尺寸）
    e_x = xdrr.get_projection_at_frame(f_chk, volume, coronary).squeeze().cpu().numpy()
    e_t = torch_drr.get_projection_at_frame(f_chk, volume, coronary).squeeze().cpu().numpy()
    ncc = _ncc(e_x, e_t)
    assert ncc > 0.98, f"pig 几何 frame {f_chk}: NCC={ncc:.4f} 过低"
    print(f"[xcist-pig] vs TorchDRR @frame {f_chk}: NCC={ncc:.4f}")

    # ── 2) 患者剂量：参考 CT 用**真实 HU 的 NIfTI**（reader 的 volume 是 μ 体积，
    #       且含 DSA 用的夸张造影 μ=0.25/mm，直接当 HU 会失真）+ 心腔 ROI ──
    n, ns = PIG_DOSE_CFG.n_voxel, PIG_DOSE_SLICES
    vxy = PIG_DOSE_CFG.fov_mm / n
    vz = PIG_DOSE_CFG.slice_thickness_mm

    hu_all, aff_all = _load_nifti(volume_dvf_dir / "image.nii.gz")
    cav_all, cav_aff = _load_nifti(volume_dvf_dir / "cavity.nii.gz")
    cor_all, _ = _load_nifti(volume_dvf_dir / "coronary.nii.gz")
    assert np.allclose(aff_all, cav_aff)
    iso = iso_from_label(cor_all, aff_all)      # 网格中心 = LCA 质心（与体模 iso 同一解剖点）

    hu_xyz = _grid_sample(hu_all, aff_all, iso, n, ns, vxy, vz, order=1, cval=-1000.0)
    hu_xyz[hu_xyz < -800] = -1000.0          # 体外/肺 → 空气（HU=0 会被当水）
    roi = _grid_sample(cav_all, cav_aff, iso, n, ns, vxy, vz, order=0, cval=0.0) > 0
    assert roi.sum() > 10, f"心脏 ROI 太小: {int(roi.sum())} voxels"
    assert np.isfinite(hu_xyz).all()
    body_frac = float((hu_xyz > -300).mean())
    assert 0.02 < body_frac < 0.9, f"体模填充比例异常: {body_frac:.3f}"

    xdrr.set_dose_reference_hu(hu_xyz)
    res = xdrr.estimate_run_dose(roi_mask=roi)
    assert res is not None
    meta, dose = res["meta"], res["dose"]
    assert meta["hu_source"] == "reference_ct"
    assert dose.shape == (n, n, ns)
    assert np.isfinite(dose).all() and dose.max() > 0.0
    stats = meta["stats"]
    for k in ("whole_mean", "whole_p99_9", "whole_max", "iso", "roi_mean", "roi_p99_9"):
        assert k in stats and np.isfinite(stats[k]), f"stats[{k}] 缺失/非有限"
    assert 0 < stats["roi_mean"] < stats["whole_max"]
    # 量级合理性闸门（0.1mm Cu / 50mA / 60 帧协议，mGy 级）
    assert 0.01 < stats["whole_mean"] < 20.0, stats
    assert stats["iso"] > 0.0
    assert stats["roi_p99_9"] > stats["roi_mean"]

    # ── 3) CTDI 口径（随 run dose 一起算出来的） ──
    assert "ctdi" in meta, "run dose 未附带 CTDI 口径"
    ctdi = meta["ctdi"]
    for k in ("ctdi_w", "ctdi_w_mean100", "mean100", "ctdi100", "per_100mAs"):
        assert k in ctdi, f"CTDI 缺少 {k}"
    assert np.isfinite(ctdi["ctdi_w"]) and ctdi["ctdi_w"] > 0
    assert np.isfinite(ctdi["ctdi_w_mean100"]) and ctdi["ctdi_w_mean100"] > 0
    assert ctdi["mean100"]["periph"] > ctdi["mean100"]["center"], "周边剂量应高于中心"
    assert ctdi["n_views_360"] > 0 and ctdi["mAs_per_rotation"] > 0
    assert ctdi["periph_fan_coverage"] == pytest.approx(1.0), ctdi["warnings"]
    # 归一化口径自洽：per_100mAs = CTDIw × 100 / mAs
    assert np.isclose(ctdi["per_100mAs"]["ctdi_w"],
                      ctdi["ctdi_w"] * 100.0 / ctdi["mAs_per_rotation"], rtol=1e-6)

    # CTDI 体模的剂量场（重算一次，用于画图；便宜的降采样版本）
    ctdi_full = xdrr.estimate_ctdi(return_dose=True)
    _save_ctdi_figure(out_dir / "ctdi_phantom.png", ctdi_full)
    assert (out_dir / "ctdi_phantom.png").stat().st_size > 5000

    # ── 4) 落盘 ──
    np.save(out_dir / "dose.npy", dose)
    dose.astype(np.float32).tofile(out_dir / "dose.mGy")
    np.save(out_dir / "dose_roi_mask.npy", roi)
    _save_dose_figure(out_dir / "dose_slices.png", dose, roi)
    assert (out_dir / "dose_slices.png").stat().st_size > 5000

    report = {
        "c_arm": xdrr.get_additional_config()["c_arm"],
        "rotate_cfg": {"total_frame": T, "fps": rcfg.fps,
                       "alpha_start": rcfg.get_rotation_angle_at_frame(0)[0],
                       "alpha_end": rcfg.get_rotation_angle_at_frame(T - 1)[0]},
        "image_ncc_vs_torch": ncc,
        "dose": meta,
        "dose_in_body": {          # 额外口径：排除体外空气后的组织平均（320mm FOV 里多数是空气）
            "mean": float(dose[hu_xyz > -300].mean()),
            "p99_9": float(np.percentile(dose[hu_xyz > -300], 99.9)),
            "n_voxel": int((hu_xyz > -300).sum()),
        },
        "ctdi": {k: v for k, v in ctdi_full.items() if k != "dose"},
        "images": [f"{out_dir}/frames/{i:03d}.png" for i in range(len(frames))],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert (out_dir / "report.json").exists()

    print(f"[xcist-pig] 患者剂量: whole mean={stats['whole_mean']:.4g} "
          f"p99.9={stats['whole_p99_9']:.4g} max={stats['whole_max']:.4g} "
          f"iso={stats['iso']:.4g} mGy | 体内组织 mean={report['dose_in_body']['mean']:.4g} "
          f"p99.9={report['dose_in_body']['p99_9']:.4g} mGy | "
          f"心脏 ROI mean={stats['roi_mean']:.4g} p99.9={stats['roi_p99_9']:.4g} mGy")
    print(f"[xcist-pig] CTDI: center={ctdi['mean100']['center']:.4g} "
          f"periph={ctdi['mean100']['periph']:.4g} -> CTDIw(mean100)={ctdi['ctdi_w_mean100']:.4g} "
          f"/ CTDIw(100/nT)={ctdi['ctdi_w']:.4g} mGy "
          f"({ctdi['per_100mAs']['ctdi_w']:.4g} mGy/100mAs, "
          f"{ctdi['n_views_360']} views/360°, {ctdi['mAs_per_rotation']:.2f} mAs/rot, "
          f"nT={ctdi['beam_width_mm']:.0f}mm)")
    print(f"[xcist-pig] 产物: {out_dir}")
