"""μ(线衰减) → xcist 材料密度体 的转换，以及多分辨率体积的公共网格工具。

用于把现有管线的 μ 体积（含造影剂增强）接入 xcist：

- 单能模式：单材料水当量密度 ρ = μ / μ_water(E) —— 在能量 E 下**精确复现** ∫μ ds。
- 多能模式：按冠脉标签/骨阈值分为 water / bone / iodine 三种材料，各材料密度
  ρ_m = μ / μ_m(E_ref)（在参考能量下复现原 μ，其它能量遵循该材料的真实衰减曲线，
  即引入 beam hardening）。

注意：xcist 中"密度图"是相对材料文件密度的缩放因子（water=1 g/cc；iodine 文件为
固态碘密度），因此 ρ_m = μ/μ_m(E_ref) 是数值一致的表达，而非严格质量浓度。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import MU_WATER
from .phantom import _sample_to_canonical

# HU 阈值（用于 μ→材料分类）
AIR_HU = -800.0
BONE_HU = 350.0


# --------------------------------------------------------------------------- 几何工具
def column_spacing(affine: np.ndarray) -> tuple[float, float, float]:
    """各体素轴的物理尺寸（mm，取 affine 前三列范数）。"""
    a = np.asarray(affine, dtype=float)[:3, :3]
    return tuple(float(np.linalg.norm(a[:, i])) for i in range(3))


def world_aabb(affine: np.ndarray, shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """体积在世界的 AABB。shape 为三个空间维度大小（对应 affine 列）。"""
    a = np.asarray(affine, dtype=float)
    n = [int(s) - 1 for s in shape]
    corners = np.array([[i, j, k, 1.0] for i in (0, n[0]) for j in (0, n[1]) for k in (0, n[2])])
    w = (a @ corners.T).T[:, :3]
    return w.min(axis=0), w.max(axis=0)


def canonical_voxel_mm(affine: np.ndarray) -> tuple[float, float, float]:
    """世界轴对齐的体素尺寸：轴对齐 affine（常见）取对角绝对值，否则退化为最小间距。"""
    a = np.asarray(affine, dtype=float)[:3, :3]
    diag = np.diag(a)
    if np.allclose(a - np.diag(diag), 0, atol=1e-6):
        return tuple(max(abs(float(diag[i])), 1e-6) for i in range(3))
    s = min(column_spacing(affine))
    return (s, s, s)


def union_canonical_grid(
    volumes: Sequence[tuple[np.ndarray, Sequence[int]]],
    voxel_mm: Optional[tuple[float, float, float]] = None,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    """由多个 (affine, shape) 求并集世界 AABB → (center, half_mm, voxel_mm)。"""
    mins, maxs = [], []
    for affine, shape in volumes:
        lo, hi = world_aabb(affine, shape)
        mins.append(lo); maxs.append(hi)
    lo = np.min(np.stack(mins), axis=0); hi = np.max(np.stack(maxs), axis=0)
    center = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-3)
    if voxel_mm is None:
        voxel_mm = canonical_voxel_mm(volumes[0][0])
    return center, tuple(float(v) for v in half), tuple(float(v) for v in voxel_mm)


def resample_scalar_to_canonical(
    data3: np.ndarray,
    affine: np.ndarray,
    center_world: Sequence[float],
    half_mm: Sequence[float],
    voxel_mm: Sequence[float],
    order: int = 1,
) -> np.ndarray:
    """(N0,N1,N2) 世界坐标体积 → 规范网格 (Nz,Ny,Nx)，ix→X, iy→Y, iz→Z。"""
    return _sample_to_canonical(
        np.asarray(data3, dtype=np.float32),
        np.asarray(affine, dtype=float),
        np.asarray(center_world, dtype=float),
        tuple(float(v) for v in half_mm),
        tuple(float(v) for v in voxel_mm),
        order=order,
    )


# --------------------------------------------------------------------------- μ → 材料
def mu_of_material_mm(material: str, kev: float) -> float:
    """材料在给定能量的线衰减系数 (mm^-1)，按其材料文件密度（GetMu 返回 cm^-1）。"""
    import gecatsim as xc
    mu_cm = float(np.asarray(xc.GetMu(material, [float(kev)])).ravel()[0])
    return mu_cm / 10.0


def mu_to_water_density(
    mu_grid: np.ndarray,
    mu_water_mm: float = MU_WATER,
) -> dict[str, np.ndarray]:
    """单材料水当量：ρ = μ/μ_water(E)。单能模式下精确复现 ∫μ ds。"""
    mu = np.asarray(mu_grid, dtype=np.float32)
    rho = np.clip(mu / max(mu_water_mm, 1e-9), 0.0, None).astype(np.float32)
    return {"water": rho}


def mu_to_materials(
    mu_grid: np.ndarray,
    *,
    coronary_mask: Optional[np.ndarray] = None,
    reference_kev: float = 70.0,
    bone_hu: float = BONE_HU,
    air_hu: float = AIR_HU,
    water_material: str = "water",
    bone_material: str = "bone",
    iodine_material: str = "I",
    mu_water_mm: float = MU_WATER,
) -> dict[str, np.ndarray]:
    """μ 体 → {water, bone, iodine} 密度体（多能模式）。

    - 组织：由 μ 反推 HU（μ=μ_w(HU/1000+1)），HU≥bone_hu → bone，否则 water；
    - 冠脉（coronary_mask>0）：iodine 材料；
    - 各材料密度 = μ/μ_mat(reference_kev)（参考能量下复现原 μ）。
    """
    mu = np.asarray(mu_grid, dtype=np.float32)
    hu = (mu / max(mu_water_mm, 1e-9) - 1.0) * 1000.0
    air_mu = mu_water_mm * (air_hu / 1000.0 + 1.0)

    iod = np.zeros_like(mu, dtype=bool)
    if coronary_mask is not None:
        iod = np.asarray(coronary_mask) > 0.5
    valid = mu > air_mu
    soft = valid & (~iod) & (hu < bone_hu)
    bone = valid & (~iod) & (hu >= bone_hu)

    out: dict[str, np.ndarray] = {}
    mu_w = float(mu_of_material_mm(water_material, reference_kev))
    w = np.where(soft, mu / max(mu_w, 1e-9), 0.0).astype(np.float32)
    if (w > 0).any():
        out[water_material] = w
    if bone.any():
        mu_b = float(mu_of_material_mm(bone_material, reference_kev))
        out[bone_material] = np.where(bone, mu / max(mu_b, 1e-9), 0.0).astype(np.float32)
    if iod.any():
        mu_i = float(mu_of_material_mm(iodine_material, reference_kev))
        out[iodine_material] = np.where(iod, mu / max(mu_i, 1e-9), 0.0).astype(np.float32)
    return out


def sum_materials(*dicts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """按材料逐元素求和（多层融合；层间互补，可直接相加）。"""
    total: dict[str, np.ndarray] = {}
    for d in dicts:
        for k, v in (d or {}).items():
            v = np.asarray(v, dtype=np.float32)
            total[k] = v if k not in total else total[k] + v
    return {k: v.astype(np.float32) for k, v in total.items() if (v > 0).any()}
