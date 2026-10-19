"""HU CTA → gecatsim 体素材料体（water/bone/air 密度 map）+ .vp/.raw 写出。

坐标约定（与 geometry.XcistGeometry 一致）：
- 世界 = 以 isocenter 为原点的 xcist 坐标（X=右, Y=前/源侧, Z=头向）。
- 规范数组用 shape (Nz, Ny, Nx) 存储：第 3 维 ix 沿世界 X(探测器 col)，
  第 2 维 iy 沿世界 Y(深度/源方向)，第 1 维 iz 沿世界 Z(探测器 row)。
- 网格中心对准 isocenter；额外摆位差通过 cfg.phantom.centerOffset 修正。

两个易错约定（均已由实验钉死，修错后 mono 与 TorchDRR 全角度 NCC 0.9999）：
1. 规范网格必须**直接**按 (z,y,x) 生成——曾用 meshgrid(x,y,z)+moveaxis 导致
   X/Y 轴互换（症状：投影出现中心竖线+水平条带、斜视角 NCC 极低）。
2. `.vp/.raw` 的**行索引与世界 Y 反向**（ix→X、iz→Z 正常），故写出时对 iy 做
   镜像，见 `FLIP_Y_IN_VP`。此镜像在 α=0/180 不可见，只在斜视角/深度位移下暴露。

标定结论（来自 temp/xcist_geom_calib.py 及 temp/xcist_single_ball.py）：
+ix→+col, +iz→+row, iy→深度（镜像后）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 材料映射默认阈值（HU）
AIR_T = -300.0     # < AIR_T → air/空（密度≈0）
BONE_T = 350.0     # >= BONE_T → bone（钙化/骨骼；CTA 中碘化血池 HU 也可能很高）

# 材料文件必须在 gecatsim/material 中存在
AIR_MAT = "air"
WATER_MAT = "water"
BONE_MAT = "bone"

#: gecatsim `Phantom_Voxelized` 的 .raw/.vp 行索引与 **世界 Y 轴反向**。
#:
#: 铁证（temp/xcist_single_ball.py，亚像素质心误差 <0.2px）：把小球放在
#: (X=0, Y=+50, Z=0)，α=+30°，xcist 测得 col=110.06，而"世界 Y 反向"模型
#: 预测 110.2、原模型预测 11.2；@Y=+25 → 87.51 vs 87.5/38.1；
#: @(25,+25,0) → 46.16 vs 46.2/-4.7。TorchDRR 则精确匹配原模型（11.14/38.05）。
#: X(col) 与 Z(row) 轴不受影响（同实验已验证）。
FLIP_Y_IN_VP = True


@dataclass
class VoxelPhantom:
    """规范网格上的多材料密度体（g/cm^3）与体素几何。"""

    arrays: dict[str, np.ndarray]   # {mat: (Nz,Ny,Nx) float32 density}
    voxel_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)  # dx,dy,dz
    name: str = "gen4d_phantom"
    hu_grid: Optional[np.ndarray] = None  # 同网格 HU (Nz,Ny,Nx)，用于参考/对比
    #: 写出时对 iy 轴做镜像（见 FLIP_Y_IN_VP）。数组本身仍是规范的 (Nz,Ny,Nx)。
    flip_y_in_vp: bool = FLIP_Y_IN_VP

    @property
    def shape(self) -> tuple[int, int, int]:
        key = next(iter(self.arrays))
        return self.arrays[key].shape

    # ---- 写出官方 .vp/.raw --------------------------------------------------
    def write(self, out_dir: str) -> str:
        """写 .vp(json)+每个材料一个 .raw，返回 json 绝对路径。

        摆位：.vp 的 offset=(dims+1)/2 使**网格中心落在等中心**(世界原点)。
        若本网格中心实际不在等中心（如体积 AABB 中心不在原点），必须把该中心
        作为 `cfg.phantom.centerOffset` 传给 gecatsim（见 `XcistSim._apply_center_offset`），
        否则整个物体会被平移（实测 NCC 0.42 vs 0.9998）。
        """
        os.makedirs(out_dir, exist_ok=True)
        Nz, Ny, Nx = self.shape
        mats, raws = [], []
        for name, vol in self.arrays.items():
            raw = os.path.join(out_dir, f"{self.name}_{name}.raw")
            arr = np.asarray(vol, dtype=np.float32)
            if self.flip_y_in_vp:
                arr = arr[:, ::-1, :]      # iy ↔ 世界 -Y（gecatsim .vp 约定）
            np.ascontiguousarray(arr, dtype=np.float32).tofile(raw)
            mats.append(name)
            raws.append(os.path.basename(raw))
        dx, dy, dz = self.voxel_mm
        vp = {
            "n_materials": len(mats),
            "mat_name": mats,
            "volumefractionmap_filename": raws,
            "volumefractionmap_datatype": ["float"] * len(mats),
            "cols": [Nx] * len(mats),
            "rows": [Ny] * len(mats),
            "slices": [Nz] * len(mats),
            "x_size": [dx] * len(mats),
            "y_size": [dy] * len(mats),
            "z_size": [dz] * len(mats),
            # offset=(dims+1)/2：网格中心对准 isocenter（经验标定）
            "x_offset": [(Nx + 1) / 2] * len(mats),
            "y_offset": [(Ny + 1) / 2] * len(mats),
            "z_offset": [(Nz + 1) / 2] * len(mats),
        }
        json_path = os.path.join(out_dir, f"{self.name}.json")
        with open(json_path, "w") as f:
            json.dump(vp, f)
        return json_path

    # ---- 统计 ---------------------------------------------------------------
    def describe(self) -> str:
        parts = [f"shape={self.shape} voxel={self.voxel_mm}"]
        for name, vol in self.arrays.items():
            parts.append(f"{name}: max={vol.max():.3f} nnz={int((vol > 0).sum())}")
        return "; ".join(parts)


def hu_to_materials(
    hu: np.ndarray,
    air_t: float = AIR_T,
    bone_t: float = BONE_T,
) -> dict[str, np.ndarray]:
    """把 HU 体 (Nz,Ny,Nx) 映射为 {water, bone} 密度体。

    软组织用水当量密度近似：rho=(HU+1000)/1000（HU=0→1.0 g/cc）；
    骨骼为独立 'bone' 材料、同样按密度缩放；空气/体外 → 0。
    """
    hu = np.asarray(hu, dtype=np.float32)
    dens = np.clip((hu + 1000.0) / 1000.0, 0.0, 3.0)
    water = np.where((hu >= air_t) & (hu < bone_t), dens, 0.0).astype(np.float32)
    bone = np.where(hu >= bone_t, dens, 0.0).astype(np.float32)
    return {WATER_MAT: water, BONE_MAT: bone}


def _sample_to_canonical(
    data: np.ndarray,
    affine: np.ndarray,
    iso_world: np.ndarray,
    half_mm: tuple[float, float, float],
    voxel_mm: tuple[float, float, float],
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """把原始数据(轴序对应 affine)重采样到以 iso 为中心的规范网格。

    返回 HU 数组 shape (Nz,Ny,Nx)：ix→X, iy→Y(源方向), iz→Z。

    `cval` 是体积外（以及 crop 外的 FOV）的填充值：
      * 密度/μ 体积 → 默认 0.0（即空气，正确）；
      * **HU 体积必须传 -1000.0**：HU=0 会被剂量/体模代码当作"水"，
        否则体积外会多出一圈虚拟水（实测会明显压低剂量）。
    """
    from scipy.ndimage import map_coordinates

    half_x, half_y, half_z = half_mm
    dx, dy, dz = voxel_mm
    Nx = int(round(2 * half_x / dx))
    Ny = int(round(2 * half_y / dy))
    Nz = int(round(2 * half_z / dz))

    x = (np.arange(Nx) - (Nx - 1) / 2.0) * dx   # world X (col)
    y = (np.arange(Ny) - (Ny - 1) / 2.0) * dy   # world Y (depth)
    z = (np.arange(Nz) - (Nz - 1) / 2.0) * dz   # world Z (row)
    # 规范数组 (Nz,Ny,Nx)：必须直接按 (z,y,x) 生成网格，
    # 否则 X/Y 轴会被互换（见 temp/xcist_mono_probe.py 的解析探针）。
    Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")  # (Nz,Ny,Nx)
    world = np.stack([Xg + iso_world[0], Yg + iso_world[1], Zg + iso_world[2]], axis=-1)

    # world → 原始 voxel
    invA = np.linalg.inv(affine)
    hom = np.concatenate([world.reshape(-1, 3), np.ones((world.size // 3, 1))], axis=1)
    v = (invA @ hom.T).T[:, :3]     # (M,3) in original voxel axes (i,j,k)
    vi, vj, vk = v[:, 0], v[:, 1], v[:, 2]
    sampled = map_coordinates(
        np.asarray(data, dtype=np.float64), [vi, vj, vk],
        order=order, mode="constant", cval=float(cval), prefilter=False,
    )
    return sampled.reshape(Nz, Ny, Nx).astype(np.float32)


def build_from_ct(
    hu: np.ndarray,
    affine: np.ndarray,
    iso_world: Optional[np.ndarray] = None,
    *,
    half_mm: tuple[float, float, float] = (95.0, 95.0, 85.0),
    voxel_mm: tuple[float, float, float] = (0.6, 0.6, 0.625),
    bone_t: float = BONE_T,
    air_t: float = AIR_T,
    name: str = "gen4d_phantom",
) -> VoxelPhantom:
    """从 NIfTI HU + affine 构建中心在 iso 的 VoxelPhantom。

    iso_world: 目标结构（冠脉）中心在原始世界的坐标；缺省取体积中心。
    """
    hu = np.asarray(hu)
    if iso_world is None:
        c = np.array(hu.shape) / 2.0
        iso_world = (affine[:3, :3] @ c + affine[:3, 3])
    iso_world = np.asarray(iso_world, dtype=float)
    grid = _sample_to_canonical(hu, affine, iso_world, half_mm, voxel_mm,
                                cval=-1000.0)   # 体积外 = 空气（HU=0 会被当水）
    grid[grid < -800] = -1000.0     # 体外
    arrays = hu_to_materials(grid, air_t=air_t, bone_t=bone_t)
    # 去掉全 0 材料以省内存
    arrays = {k: v for k, v in arrays.items() if (v > 0).any()}
    return VoxelPhantom(arrays=arrays, voxel_mm=voxel_mm, name=name,
                        hu_grid=grid)


def iso_from_label(
    label: np.ndarray, affine: np.ndarray, margin_mm: float = 0.0
) -> np.ndarray:
    """由标签前景(世界)求目标中心 iso（几何中心，Z 取标签中心）。"""
    idx = np.argwhere(np.asarray(label) > 0)          # (M,3) original voxel
    w = idx @ affine[:3, :3].T + affine[:3, 3]
    return w.mean(axis=0)
