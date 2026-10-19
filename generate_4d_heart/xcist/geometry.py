"""C-arm 几何 → gecatsim scanner/protocol 的映射与解析 oracle。

约定（本 spike 采用，可在标定后修正为实测结果）：
- xcist 世界坐标 = 以 isocenter 为原点、与 RAS 对齐（X=患者右，
  Y=前/AP 源侧，Z=头向/旋转轴）。该映射的正确性由
  `calibrate_geometry` 经验标定 + 与 Torch3D/DiffDRR 叠加验证。
- gecatsim 视角 0 时源在 +Y、探测器在 -Y，平板沿 +X(col)/+Z(row)。
- 源-探测器绕 Z 旋转角 beta（度），即 cfg.protocol.startAngle
  （单 view 下 = 该角度）。

本文件提供：
1. XcistGeometry：把 CArmGeometry/RotatedParameters 映射成 xcist
   scanner/protocol 字段（含 AP/PA、alpha→beta 映射、旋转方向）。
2. 解析 forward projector（oracle）：给定 xcist 世界点 + alpha，
   预测其落在探测器图像的行/列（用于标定与单元测试），与
   `convert_json_to_proj_matrics` 的物理一致。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# 可标定常量
#
# 标定状态（2026-10-14 重新验证，修正 phantom 两个约定 bug 后）：
#   - 解析探针 `temp/xcist_mono_probe.py`：已知几何体模的特征位置与
#     `project_point` 预测一致（质心 <0.2px）。
#   - 真实 CTA 全角度对比 `temp/xcist_drr_validate.py`：vs TorchDRR/DiffDRR
#     NCC = 0.9998~0.9999（α ∈ {0,±30,60,90,135,150,180}），argmax 逐像素重合。
#   - 模型：源在 (0, SID, 0)、绕 +Z 旋转 beta=startAngle，探测器 (0, -DID, 0)，
#     u=+X(col)、v=+Z(row)（已验证 cfg 中 srcNew 坐标与该公式精确一致）。
# 注意：体模文件侧还有一个独立的**Y 镜像**约定，见 phantom.FLIP_Y_IN_VP。
# ---------------------------------------------------------------------------
#: 读回图像行方向翻转。1=不翻, -1=翻。
FLIP_ROW = -1
#: 读回图像列方向翻转。1=不翻, -1=翻。
FLIP_COL = -1


@dataclass
class XcistGeometry:
    """描述 gecatsim 平板扫描几何与坐标系映射参数。"""

    sid: float            # 源到等中心 mm（= CArm.sod）
    sdd: float            # 源到探测器 mm（= CArm.sdd）
    n_col: int            # 探测器列数（沿平板横向 u）
    n_row: int            # 探测器行数（沿平板纵向 v）
    col_size: float       # 列像素尺寸 mm
    row_size: float       # 行像素尺寸 mm
    # 坐标/角度映射参数
    orientation: str = "AP"     # AP / PA
    #: alpha(C-arm 主角, 绕 RAS Z, AP 时源在 +Y) → gecatsim beta 的换算
    alpha_to_beta: float = 1.0   # beta = beta0 + alpha_to_beta * alpha
    beta0: float = 0.0           # AP=0; PA=180（源翻到 -Y 侧）
    rotation_direction: int = 1  # gecatsim rotationDirection
    flip_row: int = FLIP_ROW
    flip_col: int = FLIP_COL

    # --- 由 CArmGeometry 构造 -------------------------------------------------
    @classmethod
    def from_c_arm(cls, c_arm, *, width_as_col: bool = True) -> "XcistGeometry":
        """从项目 CArmGeometry 构造。

        约定探测器列=图像横向(width, delx)、行=图像纵向(height, dely)。
        """
        if width_as_col:
            n_col, col_size = c_arm.width, c_arm.delx
            n_row, row_size = c_arm.height, c_arm.dely
        else:
            n_row, row_size = c_arm.width, c_arm.delx
            n_col, col_size = c_arm.height, c_arm.dely
        orientation = c_arm.orientation_type
        beta0 = 0.0 if orientation == "AP" else 180.0
        return cls(
            sid=c_arm.sod, sdd=c_arm.sdd,
            n_col=n_col, n_row=n_row,
            col_size=col_size, row_size=row_size,
            orientation=orientation, beta0=beta0,
        )

    # --- 填充 gecatsim cfg ----------------------------------------------------
    def apply_to_cfg(self, cfg) -> None:
        """把几何写进已初始化的 CatSim cfg（scanner + 部分 physics）。"""
        s = cfg.scanner
        s.sid = float(self.sid)
        s.sdd = float(self.sdd)
        s.detectorCallback = "Detector_FlatPanel"
        s.detectorColCount = int(self.n_col)
        s.detectorRowCount = int(self.n_row)
        s.detectorRowsPerMod = int(self.n_row)   # flat panel: rows per mod = rows
        s.detectorColSize = float(self.col_size)
        s.detectorRowSize = float(self.row_size)
        s.detectorColOffset = 0.0
        s.detectorRowOffset = 0.0
        s.detectorColFillFraction = 1.0
        s.detectorRowFillFraction = 1.0
        # 平板探测器需要在每像素 1:1 采样（后续可开 oversample 提画质）
        cfg.physics.FlatPanel_OSfactor = 1.0

    def alpha_to_start_angle(self, alpha_deg: float) -> float:
        """单 view 投影：alpha(deg) → cfg.protocol.startAngle(deg)。"""
        return self.beta0 + self.alpha_to_beta * alpha_deg

    def apply_frame_to_cfg(self, cfg, alpha_deg: float) -> None:
        """把某一帧的角度写到 protocol（单 view 扫描用）。"""
        cfg.protocol.startAngle = float(self.alpha_to_start_angle(alpha_deg))
        cfg.protocol.rotationDirection = int(self.rotation_direction)
        cfg.protocol.viewCount = 1
        cfg.protocol.viewsPerRotation = 360  # 步进无关：单 view 用 startAngle
        cfg.protocol.startViewId = 0
        cfg.protocol.stopViewId = 0

    # --- 解析 oracle：世界点 → 图像 (row, col) --------------------------------
    def project_point(self, world_xyz_mm, alpha_deg: float):
        """预测 xcist 世界点投影到的图像 (row, col)（0 基，含 flip 后）。

        物理模型：源在 (0, SID, 0)，绕 +Z 转 beta；探测器中心 (0,-DID,0)，
        平板法向 +Y(指向源)，u=+X(col), v=+Z(row)。
        """
        beta = math.radians(self.alpha_to_start_angle(alpha_deg))
        c, s = math.cos(beta), math.sin(beta)
        p = np.asarray(world_xyz_mm, dtype=float).reshape(3)
        # 转到"源在 +Y"的静止系：绕 Z 转 -beta
        x = c * p[0] + s * p[1]
        y = -s * p[0] + c * p[1]
        z = p[2]
        # 透视到探测器平面
        t = self.sdd / (self.sid - y)          # 放大比
        xu = x * t                             # 探测器平面横向位置 mm
        zv = z * t
        col_f = xu / self.col_size + (self.n_col - 1) / 2.0
        row_f = zv / self.row_size + (self.n_row - 1) / 2.0
        # 应用读回翻转
        if self.flip_col < 0:
            col_f = (self.n_col - 1) - col_f
        if self.flip_row < 0:
            row_f = (self.n_row - 1) - row_f
        return float(row_f), float(col_f)


@dataclass
class XcistScanProtocol:
    """旋转采集协议（真实 C-arm 参数）。

    默认对应真实扫描：90kV / 50mA / 0.1mm Cu / 每帧 3.3ms；60 fps、75°/s、持续 3s。
    → 总线数 = fps×duration = 180，角步 = 75/60 = 1.25°，总角度 = 75×3 = 225°。
    """

    alpha_start_deg: float = -112.5      # 起始主角（默认让 225° 扫描大致居中于 0°）
    angular_velocity_deg_s: float = 75.0
    fps: float = 60.0
    duration_s: float = 3.0
    include_end: bool = False            # False: 从 t=0 起算每帧，末尾不重复

    @property
    def total_frames(self) -> int:
        n = int(round(self.fps * self.duration_s))
        return n + 1 if self.include_end else n

    @property
    def angular_step_deg(self) -> float:
        return self.angular_velocity_deg_s / self.fps

    @property
    def angular_range_deg(self) -> float:
        return self.angular_velocity_deg_s * self.duration_s

    def alphas(self) -> np.ndarray:
        """逐帧主角 alpha（度）。默认 alpha_f = start + f·step。"""
        n = int(round(self.fps * self.duration_s))
        a = self.alpha_start_deg + np.arange(n) * self.angular_step_deg
        if self.include_end:
            a = np.append(a, self.alpha_start_deg + self.angular_range_deg)
        return a

    def to_dict(self) -> dict:
        return {
            "alpha_start_deg": self.alpha_start_deg,
            "angular_velocity_deg_s": self.angular_velocity_deg_s,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "total_frames": self.total_frames,
            "angular_step_deg": self.angular_step_deg,
            "angular_range_deg": self.angular_range_deg,
            "alpha_end_deg": self.alpha_start_deg + self.angular_range_deg,
        }
