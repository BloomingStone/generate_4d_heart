"""评估：我的 Cu 过滤衰减为何比设备弱？——分解为「谱(基线滤过)」vs「滤过实现」。

1. 用 CatSim 自己的 μ 表和谱文件做独立手算，验证 `catsimfilter` 的实现是否正确；
2. 与实测（设备剂量报告 0/0.1/0.2/0.3 Cu 的比值）对比；
3. 反推：需要多少额外基线滤过（Al/Cu）才能把我的手算曲线移到实测曲线上。

用法：pixi run python scripts/xcist/cu_filter_eval.py
"""
from __future__ import annotations

import numpy as np

from generate_4d_heart.xcist.dose import air_mu_en_rho
from gecatsim.pyfiles.GetMu import GetMu
from gecatsim.pyfiles.Spectrum import spectrum_read

SPEC_DIR = ".pixi/envs/default/lib/python3.14/site-packages/gecatsim/spectrum"


def load(kv: int = 90):
    e, i, _ = spectrum_read(f"{SPEC_DIR}/tungsten_tar7.0_{kv}_filt.dat")
    return np.asarray(e, float), np.asarray(i, float)


def trans_curve(e, phi, material: str, thicknesses_mm):
    """按空气比释动能加权的透射比曲线（材料厚度 mm）。"""
    mu = np.asarray(GetMu(material, e), float).ravel()      # 1/cm
    w = phi * e * air_mu_en_rho(e)                          # kerma 权重
    out = []
    for t in thicknesses_mm:
        out.append(float((w * np.exp(-mu * t * 0.1)).sum() / w.sum()))
    return np.array(out)


def equivalent_thickness(e, phi, material, target, lo=0.0, hi=3.0):
    """反推：透射比达到 target 所需的材料厚度 (mm)。"""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if trans_curve(e, phi, material, [mid])[0] > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    e, phi = load(90)
    print(f"谱: tungsten_tar7.0_90_filt.dat  {e.size} bins, {e[0]:.2f}~{e[-1]:.2f} keV")
    print(f"  注量平均能量 = {(e*phi).sum()/phi.sum():.2f} keV, "
          f"比释动能平均能量 = {(e*phi*e).sum()/(phi*e).sum():.2f} keV\n")

    ts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    cu = trans_curve(e, phi, "cu", ts)
    print("CatSim(μ表 + 谱) 手算的 Cu 透射比（空气比释动能加权）:")
    for t, v in zip(ts, cu):
        print(f"   {t:.2f} mm Cu -> {v:.3f}")

    # 我实际跑出来的 Ka,r（187 帧 50 mA 0.1/0.2 Cu，见 scripts/xcist/dose_vs_measured.py）
    mine = {0.0: 1.990, 0.1: 1.313, 0.2: 0.956}
    print("\n我实测(gecatsim 端到端) 相对 0Cu 的比值:", end=" ")
    print({k: round(v / mine[0.0], 3) for k, v in mine.items()})

    # 设备剂量报告（90kV，同一行内只有 0/0.1 @140mA，0.1/0.2 @50mA）
    print("\n设备报告：")
    print(f"   0Cu/0.1Cu @140mA: {16.373:.3f} / {8.802:.3f} = {8.802/16.373:.3f}")
    print(f"   0.1/0.2Cu @50mA : {3.915:.3f} / {2.652:.3f} = {2.652/3.915:.3f}")
    print("   (注: 两行的 mA 不同，只能看'加厚一层 Cu 的相对削减'，不能跨行比绝对值)")

    print("\n== 若要让我的手算曲线复现实测的 0.1Cu 透射比 0.538 ==")
    t_eq = equivalent_thickness(e, phi, "cu", 0.538)
    print(f"  需要 ~{t_eq:.3f} mm Cu 才能把透射比压到 0.538（标称只有 0.1 mm）")
    print("  → 说明设备的 0Cu 基线谱比我的软得多，或标称 0.1Cu 实际更厚/含其它滤过")

    print("\n== 给基线加固有滤过 Al 后再算 0.1Cu 透射比（看能否对上 0.538）==")
    for al in (0.0, 1.0, 2.0, 3.0, 4.0):
        phi2 = phi * np.exp(-np.asarray(GetMu("al", e), float).ravel() * al * 0.1)
        t01 = trans_curve(e, phi2, "cu", [0.1])[0]
        t02 = trans_curve(e, phi2, "cu", [0.2])[0]
        print(f"   +{al:.0f}mm Al 基线: 0.1Cu 透射比 = {t01:.3f} "
              f"(0.1→0.2 增量 = {t02/t01:.3f})")


if __name__ == "__main__":
    main()
