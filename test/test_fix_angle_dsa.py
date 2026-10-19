"""固定视角（single-angle）DSA 的单元 + 端到端测试。

参考 `dev/single-angle` 分支的实现移植，但改用当前 API：
固定视角 = `angular_velocity=0` + `alpha_start/beta_start`，整条 `RotateDSA`
流水线原样复用（含静态层批量、标签/深度、可选剂量）。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from generate_4d_heart.rotate_dsa.fix_angle_dsa import (
    FixAngleDSA, fix_angle_rotate_cfg, format_angle, parse_angle,
)
from generate_4d_heart.rotate_dsa.rotate_drr import (
    CArmGeometry, RotatedParameters, TorchDRR,
)

from utils import ReaderName, SimulatorName, reader_factory

#: 小几何：128²@1mm（等中心 FOV ≈ 65mm），3 帧足够验证"固定视角"
FIX_GEOM = CArmGeometry(sdd=1200.0, sod=760.0, height=128, width=128,
                        delx=1.0, dely=1.0, orientation_type="AP")
FIX_FRAMES = 3
FIX_FPS = 10.0


@pytest.mark.parametrize("text,expected", [
    ("AP", (0.0, 0.0)),
    ("LAT", (90.0, 0.0)),
    ("RLAT", (-90.0, 0.0)),
    ("RAO_30", (30.0, 0.0)),
    ("LAO_60", (-60.0, 0.0)),
    ("CRAN_20", (0.0, 20.0)),
    ("CAUD_10", (0.0, -10.0)),
    ("RAO_30_CRAN_20", (30.0, 20.0)),
    ("LAO_60_CAUD_10", (-60.0, -10.0)),
    ("RAO30", (30.0, 0.0)),          # 无下划线写法
    ("30,10", (30.0, 10.0)),
    ("30 10", (30.0, 10.0)),
    ("30", (30.0, 0.0)),
])
def test_parse_angle(text: str, expected: tuple[float, float]):
    assert parse_angle(text) == expected


def test_parse_angle_invalid():
    with pytest.raises(ValueError):
        parse_angle("XX_10")


@pytest.mark.parametrize("alpha,beta,name", [
    (0.0, 0.0, "AP"),
    (90.0, 0.0, "LAT"),
    (30.0, 0.0, "RAO_30"),
    (-60.0, 0.0, "LAO_60"),
    (30.0, 20.0, "RAO_30_CRAN_20"),
    (0.0, -10.0, "CAUD_10"),
])
def test_format_angle(alpha: float, beta: float, name: str):
    assert format_angle(alpha, beta) == name
    # 往返一致
    assert parse_angle(format_angle(alpha, beta)) == pytest.approx((alpha, beta))


def test_fix_angle_rotate_cfg():
    base = RotatedParameters(total_frame=7, fps=15.0, alpha_start=-120.0,
                             beta_start=5.0, angular_velocity=75.0)
    cfg = fix_angle_rotate_cfg(base, 30.0, 20.0)
    assert cfg.angular_velocity == 0.0
    assert (cfg.alpha_start, cfg.beta_start) == (30.0, 20.0)
    # 帧数/fps 等其它字段保持不变
    assert (cfg.total_frame, cfg.fps) == (7, 15.0)
    assert base.angular_velocity == 75.0, "不应原地修改输入"
    # 所有帧角度相同
    angs = {cfg.get_rotation_angle_at_frame(f) for f in range(cfg.total_frame)}
    assert len(angs) == 1


@pytest.fixture(scope="module")
def fix_reader():
    _, reader = reader_factory(
        SimulatorName.STATIC_IODINE_CONTRAST,
        ReaderName.RBF_READER,
        cororonary_need_load="LCA",
    )
    yield reader
    from generate_4d_heart.rotate_dsa.data_reader.data_reader import cleanup_gpu_memory

    cleanup_gpu_memory()


@pytest.fixture(scope="module")
def fix_drr() -> TorchDRR:
    # 注意：这里故意给一个"旋转协议"，由 FixAngleDSA 改成固定视角
    return TorchDRR(
        c_arm_cfg=FIX_GEOM,
        rotate_cfg=RotatedParameters(
            total_frame=FIX_FRAMES, fps=FIX_FPS,
            alpha_start=-120.0, beta_start=0.0, angular_velocity=60.0),
        patch_size=128,
    )


def test_fix_angle_dsa_runs_at_single_view(
    output_root_dir: Path,
    fix_reader,
    fix_drr: TorchDRR,
):
    """固定视角：所有帧同一视角、心相位推进，产物命名与 json 正确。"""
    alpha, beta = 30.0, 20.0
    dsa = FixAngleDSA(reader=fix_reader, drr=fix_drr, alpha=alpha, beta=beta)

    # 构造时已把 rotate_cfg 改成固定视角
    assert dsa.drr.rotate_cfg.angular_velocity == 0.0
    assert (dsa.drr.rotate_cfg.alpha_start, dsa.drr.rotate_cfg.beta_start) == (alpha, beta)
    assert dsa.preset_name == "RAO_30_CRAN_20"

    out_dir = output_root_dir / "fix_angle_dsa" / "rao30_cran20"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    output = dsa.run_and_save(out_dir, coronary_type="LCA", gif_fps=5)

    # ── 数值产物 ──
    T = FIX_FRAMES
    assert output.raw_frames.shape == (T, FIX_GEOM.height, FIX_GEOM.width)
    assert np.isfinite(output.raw_frames).all()
    assert 0.0 <= output.raw_frames.min() and output.raw_frames.max() <= 1.0 + 1e-6
    assert (output.labels > 0).any(), "labels 全为 0"
    assert np.isfinite(output.depth_maps).any(), "depth 全为 nan"
    # 固定视角下，帧间变化应来自心动/造影（幅度小而存在）
    assert float(np.abs(output.raw_frames[0] - output.raw_frames[-1]).mean()) > 1e-6

    # ── 几何 json：每帧角度恒定 + fixed_angle 段 ──
    geo = dsa.get_geometry_json("LCA")
    assert geo["fixed_angle"]["alpha_degree"] == alpha
    assert geo["fixed_angle"]["beta_degree"] == beta
    assert geo["fixed_angle"]["preset"] == "RAO_30_CRAN_20"
    assert len(geo["frames"]) == T
    alphas = {f["alpha_degree"] for f in geo["frames"]}
    betas = {f["beta_degree"] for f in geo["frames"]}
    assert alphas == {alpha} and betas == {beta}, (alphas, betas)

    # ── 落盘 ──
    assert (out_dir / "fix_angle_dsa_raw.tif").exists()
    assert (out_dir / "fix_angle_dsa.gif").exists()
    assert (out_dir / "fix_angle_dsa.json").exists()
    assert (out_dir / "label.npz").exists()
    assert (out_dir / "depth_map.npz").exists()

    saved = json.loads((out_dir / "fix_angle_dsa.json").read_text())
    assert saved["fixed_angle"]["preset"] == "RAO_30_CRAN_20"
    assert len(saved["frames"]) == T
    assert {f["alpha_degree"] for f in saved["frames"]} == {alpha}

    print(f"[fix-angle] {dsa.preset_name}: raw{output.raw_frames.shape} "
          f"labels 非零 {int((output.labels > 0).sum())} px -> {out_dir}")


def test_fix_angle_dsa_from_angle(fix_reader, fix_drr: TorchDRR):
    """`from_angle` 工厂：字符串 → (alpha, beta)，并默认写 `fix_angle_dsa*` 前缀。"""
    dsa = FixAngleDSA.from_angle(fix_reader, fix_drr, "LAO_45_CAUD_15")
    assert (dsa.alpha, dsa.beta) == (-45.0, -15.0)
    assert dsa.preset_name == "LAO_45_CAUD_15"
    assert dsa.drr.rotate_cfg.angular_velocity == 0.0
