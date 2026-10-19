"""固定视角（single-angle）DSA 投影。

与 `RotateDSA`（旋转采集）相对：C 臂固定在某个 C-arm 角度 ``(alpha, beta)``，
采集期间只有**心脏相位/造影剂**随时间演化。

思路
----
`RotateDRR` 的视角完全由 `rotate_cfg` 决定，因此"固定视角" = 把
`RotatedParameters.angular_velocity` 置 0、并把 `alpha_start/beta_start` 设为目标角度。
这样所有帧的 `get_rotation_angle_at_frame(f)` 都返回同一对角度，
整条 `RotateDSA` 流水线（静态层批量、逐帧动态层、标签/深度、伪影、剂量）**原样复用**，
无需另写一份 `run()`。

角度约定（与 `RotatedParameters` 一致，RAS 坐标）
------------------------------------------------
- ``alpha``：绕 SI(Z) 轴，**正值为 RAO（右前斜）**、负值为 LAO（左前斜）
- ``beta``：绕 RL(X) 轴，**正值为 CRAN（头位）**、负值为 CAUD（足位）
- ``AP`` = (0, 0)；``LAT/LLAT`` = (90, 0)；``RLAT`` = (-90, 0)

用法
----
>>> from generate_4d_heart.rotate_dsa.fix_angle_dsa import FixAngleDSA, parse_angle
>>> alpha, beta = parse_angle("RAO_30_CRAN_20")        # → (30.0, 20.0)
>>> dsa = FixAngleDSA.from_angle(reader=reader, drr=torch_drr, angle="RAO_30")
>>> dsa.run_and_save(output_dir, coronary_type="LCA")  # → fix_angle_dsa*.{tif,gif,json}

CLI 见 `scripts/fix_angle_dsa.py`。
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from ..types import Degree
from .data_reader import DataReader
from .rotate_drr import RotateDRR, RotatedParameters
from .rotate_dsa import PhysicalConfig, RotateDSA, RotateDsaOutput

#: 常用 C 臂预设角度 (alpha, beta)，单位度
PRESET_ANGLES: dict[str, tuple[float, float]] = {
    "AP": (0.0, 0.0),       # 正位
    "LAT": (90.0, 0.0),     # 侧位（左）
    "LLAT": (90.0, 0.0),    # 左侧位
    "RLAT": (-90.0, 0.0),   # 右侧位
}

_DIRECTIONS = ("RAO", "LAO", "CRAN", "CAUD")


def parse_angle(angle_str: str) -> tuple[float, float]:
    """把角度字符串解析为 ``(alpha, beta)``（度）。

    支持：
    - 直接数值：``"30,10"`` / ``"30 10"`` / ``"30"``（后者 beta=0）
    - 预设名：``"AP"`` / ``"LAT"`` / ``"LLAT"`` / ``"RLAT"``
    - 方向写法：``"RAO_30"``（alpha=+30）、``"LAO_60"``（alpha=−60）、
      ``"CRAN_20"``（beta=+20）、``"CAUD_10"``（beta=−10）
    - 组合：``"RAO_30_CRAN_20"``、``"LAO_60_CAUD_10"``（下划线可省：``"RAO30"``）
    """
    s = angle_str.strip()
    if not s:
        raise ValueError("角度字符串为空")

    # 1) "alpha,beta" / "alpha beta" / 单独 alpha
    if re.fullmatch(r"-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?", s):
        a, b = s.split(",")
        return (float(a), float(b))
    if re.fullmatch(r"-?\d+(\.\d+)?\s+-?\d+(\.\d+)?", s):
        a, b = s.split()
        return (float(a), float(b))
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return (float(s), 0.0)

    # 2) 预设名
    upper = s.upper()
    if upper in PRESET_ANGLES:
        return PRESET_ANGLES[upper]

    # 3) RAO/LAO/CRAN/CAUD（方向本身决定正负号）
    alpha = beta = 0.0
    found = False
    for direction, deg in re.findall(r"(RAO|LAO|CRAN|CAUD)_?(\d+(?:\.\d+)?)", upper):
        found = True
        v = float(deg)
        if direction == "RAO":
            alpha += v
        elif direction == "LAO":
            alpha -= v
        elif direction == "CRAN":
            beta += v
        else:                      # CAUD
            beta -= v
    if found:
        return (alpha, beta)

    raise ValueError(
        f"无法解析角度字符串: {angle_str!r}；支持 "
        f"{list(PRESET_ANGLES)}、'RAO_30'、'LAO_60'、'CRAN_20'、'CAUD_10'、"
        "'RAO_30_CRAN_20'、'30,10'、'30 10'、'30'"
    )


def format_angle(alpha: Degree, beta: Degree) -> str:
    """``(alpha, beta)`` → 惯用角度名（如 ``"RAO_30_CRAN_20"``、``"AP"``）。"""
    al, be = float(alpha), float(beta)
    for name, (a, b) in PRESET_ANGLES.items():
        if abs(al - a) < 1e-6 and abs(be - b) < 1e-6:
            return name
    parts: list[str] = []
    if abs(al) > 1e-6:
        parts.append(f"{'RAO' if al > 0 else 'LAO'}_{abs(al):g}")
    if abs(be) > 1e-6:
        parts.append(f"{'CRAN' if be > 0 else 'CAUD'}_{abs(be):g}")
    return "_".join(parts) if parts else "AP"


def fix_angle_rotate_cfg(
    base: RotatedParameters,
    alpha: Degree,
    beta: Degree,
) -> RotatedParameters:
    """把旋转协议改成"固定视角"：保留帧数/fps，置角速度 0 并设定 alpha/beta。"""
    return replace(
        base,
        alpha_start=float(alpha),
        beta_start=float(beta),
        angular_velocity=0.0,
    )


@dataclass
class FixAngleDSA:
    """固定视角 DSA：`RotateDSA` 的薄封装（视角固定、心相位推进）。

    Parameters
    ----------
    reader : DataReader
        数据读取器（HU→μ/造影剂 由 reader 内部的 ContrastSimulator 完成）。
    drr : RotateDRR
        投影后端（`TorchDRR` 或 `XcistDRR`）。
    alpha, beta : float
        固定角度（度）；alpha RAO 为正、beta CRAN 为正。
    physical_config : PhysicalConfig
        心周期等物理参数。
    multi_threads : bool
        是否启用 `RotateDSA` 的生产者线程。

    Notes
    -----
    构造时会**改写** `drr.rotate_cfg`（`angular_velocity=0` 且 alpha/beta 置为给定值），
    因为视角完全由该对象决定。若同一个 `drr` 还要用于旋转采集，请先备份 `rotate_cfg`。
    """

    reader: DataReader
    drr: RotateDRR
    alpha: Degree = 0.0
    beta: Degree = 0.0
    physical_config: PhysicalConfig = field(default_factory=PhysicalConfig)
    multi_threads: bool = True
    _dsa: RotateDSA = field(init=False, repr=False)

    def __post_init__(self) -> None:
        prev = self.drr.rotate_cfg
        if abs(float(prev.angular_velocity)) > 1e-9:
            warnings.warn(
                f"FixAngleDSA: 将 drr.rotate_cfg.angular_velocity 由 {prev.angular_velocity} "
                "改为 0（固定视角），并改写 alpha_start/beta_start；"
                "若同一 drr 还要做旋转采集，请先备份 rotate_cfg",
                stacklevel=2,
            )
        self.drr.rotate_cfg = fix_angle_rotate_cfg(prev, self.alpha, self.beta)
        self._dsa = RotateDSA(
            reader=self.reader,
            drr=self.drr,
            physical_config=self.physical_config,
            multi_threads=self.multi_threads,
        )

    # ------------------------------------------------------------------ 工厂
    @classmethod
    def from_angle(
        cls,
        reader: DataReader,
        drr: RotateDRR,
        angle: str,
        **kwargs: Any,
    ) -> "FixAngleDSA":
        """用角度字符串构造，例如 ``FixAngleDSA.from_angle(reader, drr, "RAO_30_CRAN_20")``。"""
        alpha, beta = parse_angle(angle)
        return cls(reader=reader, drr=drr, alpha=alpha, beta=beta, **kwargs)

    # ------------------------------------------------------------------ 信息
    @property
    def preset_name(self) -> str:
        return format_angle(self.alpha, self.beta)

    def angle_dict(self) -> dict[str, Any]:
        """写入 json 的固定角度信息。"""
        return {
            "alpha_degree": float(self.alpha),
            "beta_degree": float(self.beta),
            "preset": self.preset_name,
            "angular_velocity": 0.0,
        }

    # ------------------------------------------------------------------ 执行
    def run(self, coronary_type: Literal["LCA", "RCA"] = "LCA") -> RotateDsaOutput:
        return self._dsa.run(coronary_type)

    def run_and_save(
        self,
        output_dir: Path,
        coronary_type: Literal["LCA", "RCA"] = "LCA",
        gif_fps: int = 30,
        **kwargs: Any,
    ) -> RotateDsaOutput:
        """跑并落盘；主产物命名为 `fix_angle_dsa*`，json 额外带 `fixed_angle`。"""
        return self._dsa.run_and_save(
            output_dir=output_dir,
            coronary_type=coronary_type,
            gif_fps=gif_fps,
            prefix="fix_angle_dsa",
            extra_json={"fixed_angle": self.angle_dict()},
            **kwargs,
        )

    def get_geometry_json(self, coronary_type: Literal["LCA", "RCA"] = "LCA") -> dict:
        """与 `RotateDSA.get_geometry_json` 相同，另加 `fixed_angle` 段。"""
        res = self._dsa.get_geometry_json(coronary_type)
        res["fixed_angle"] = self.angle_dict()
        return res
