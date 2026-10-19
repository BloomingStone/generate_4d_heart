"""对齐设备《剂量统计》表：同一曝光参数下 我的仿真 vs 电离室 vs 设备剂量报告。

设备协议（表头）：3s 3D DSA，200°、187 帧、60 fps、90kV。
只跑 16 cm 体模那几行有电离室读数的组合（其余 mA 行按 mAs 线性外推即可）。

  A. **Ka,r**（参考点空气比释动能）：`XcistSim.spectrum()` + `dose.estimate_air_kerma`
  B. **CTDI_16 实拍累积**（本协议 200°）：`estimate_ctdi_like` 每 100 mAs × 总 mAs
  C. **CTDI_16 @360° 归一**（= B × 360/200），即"每次完整旋转的剂量指数"

用法：pixi run python scripts/xcist/dose_vs_measured.py
"""
from __future__ import annotations

from generate_4d_heart.xcist.dose import (
    CtdiConfig, estimate_air_kerma, estimate_ctdi_like,
)
from generate_4d_heart.xcist.geometry import XcistGeometry
from generate_4d_heart.xcist.sim import XcistPhysics, XcistSim

FRAMES, ARC_DEG = 187, 200.0
SID, SDD = 760.0, 1200.0
N_COL, N_ROW, PIX = 648, 474, 0.616
GEO = XcistGeometry(sid=SID, sdd=SDD, n_col=N_COL, n_row=N_ROW,
                    col_size=PIX, row_size=PIX, orientation="AP")
#: 平板 ÷4（面板尺寸不变）→ CTDI 计算快 16 倍，通量/束宽不变
GEO_FAST = XcistGeometry(sid=SID, sdd=SDD, n_col=N_COL // 4, n_row=N_ROW // 4,
                         col_size=PIX * 4, row_size=PIX * 4, orientation="AP")

_C: dict = {}


def _phys(kv: int, cu: float, mA: float, ms: float, nE: int) -> XcistPhysics:
    return XcistPhysics(spectrum_filename=f"tungsten_tar7.0_{kv}_filt.dat",
                        flat_filter_mm=([] if cu <= 0 else ["cu", cu]),
                        mA=mA, exposure_s=ms * 1e-3, fps=60.0, energy_count=nE)


def ka_r(kv: int, cu: float, mA: float, ms: float) -> dict:
    key = ("k", kv, cu, mA, ms)
    if key not in _C:
        sim = XcistSim(geometry=GEO, physics=_phys(kv, cu, mA, ms, 40))
        e, net = sim.spectrum()
        _C[key] = estimate_air_kerma(
            e, net, sid_mm=SID, sdd_mm=SDD, n_col=N_COL, n_row=N_ROW,
            col_size_mm=PIX, row_size_mm=PIX, n_frames=FRAMES,
            mAs_total=mA * ms * 1e-3 * FRAMES)
    return _C[key]


def ctdi16_per100(kv: int, cu: float) -> float:
    key = ("c", kv, cu)
    if key not in _C:
        sim = XcistSim(geometry=GEO_FAST, physics=_phys(kv, cu, 50.0, 3.3, 2))
        c = CtdiConfig(n_voxel=48, slice_thickness_mm=4.0, phantom_diameter_mm=160.0,
                       detector_downsample=1, max_views=36)
        c.view_density_per_360 = FRAMES / ARC_DEG
        r = estimate_ctdi_like(sim._ct.cfg, c,
                               mAs_per_rotation=50.0 * 3.3e-3 * c.n_views())
        _C[key] = r["per_100mAs"]["ctdi_w_mean"]
    return _C[key]


CASES = [
    # kV, Cu, mA, ms, 剂量报告, 电离室(CTDI16 实拍)
    (90, 0.0, 50.0, 3.29, None, None),
    (90, 0.1, 50.0, 3.29, 3.915, 1.7963),
    (90, 0.2, 50.0, 3.29, 2.652, None),
    (90, 0.1, 60.0, 3.30, 4.449, 2.0400),
]

print(f"协议 {FRAMES} 帧 / {ARC_DEG}° / 60fps;  SID={SID} SDD={SDD} "
      f"面板 {N_COL}x{N_ROW}@{PIX}mm")
hdr = (f"{'参数':22s} {'mAs':>6s} {'Ka,r':>7s} | {'CTDI16':>7s} {'电离室':>7s} {'差':>7s} | "
       f"{'CTDI@360':>8s} {'报告':>7s} {'报告/360':>8s} {'报告/Ka,r':>9s}")
print(hdr)
print("-" * len(hdr))
for kv, cu, mA, ms, rep, dos in CASES:
    k = ka_r(kv, cu, mA, ms)
    mAs = mA * ms * 1e-3 * FRAMES
    acc = ctdi16_per100(kv, cu) * mAs / 100.0
    acc360 = acc * 360.0 / ARC_DEG
    rep_s = f"{rep:7.3f}" if rep else f"{'-':>7s}"
    dos_s = f"{dos:7.3f}" if dos else f"{'-':>7s}"
    rat_s = f"{(acc / dos - 1) * 100:+6.1f}%" if dos else f"{'-':>7s}"
    print(f"{f'{kv}kV {cu}Cu {mA:.0f}mA {ms}ms':22s} {mAs:6.2f} {k['k_ref_total']:7.3f} | "
          f"{acc:7.3f} {dos_s} {rat_s} | {acc360:8.3f} {rep_s} "
          f"{(rep / acc360 if rep else float('nan')):8.2f} "
          f"{(rep / k['k_ref_total'] if rep else float('nan')):9.2f}")

print("\n参考：设备『剂量报告/电离室』比值 = 2.18（50/60 mA 两行完全一致）；"
      "2.18 ≈ (360/200) × 1.21")
