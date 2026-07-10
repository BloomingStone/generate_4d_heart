from __future__ import annotations
from typing import TypeVar, Generic

import torch
import numpy as np

from . rotate_dsa.types import CoronaryType


# ─── 坐标系定义 ───

class CoordinateSystem:
    """坐标系基类。

    变体:
        World       — 世界坐标系，原点在 (0,0,0)
        Centering   — 中心化坐标系，原点 = 目标中心在 World 中的位置（通过 cor_type 区分 LCA/RCA）
    """
    name: str
    translation: np.ndarray
    """该坐标系原点在 World 中的位置 (3,)。World 为 None。"""

    def __init__(self, name: str):
        self.name = name

    def transform_to(self, target: CoordinateSystem) -> np.ndarray:
        """返回 4×4 纯平移矩阵: self → target。

        原理：所有坐标系共享相同的 B（旋转+缩放），只差平移。
            p_target = p_self + (translation_target - translation_self)
        """
        if self is target or self.name == target.name:
            return np.eye(4)

        t_src = self.translation if self.translation is not None else np.zeros(3)
        t_tgt = target.translation if target.translation is not None else np.zeros(3)
        delta = t_tgt - t_src

        T = np.eye(4)
        T[:3, 3] = delta
        return T


class World(CoordinateSystem):
    """世界坐标系，原点在 (0,0,0)，无平移。"""
    def __init__(self):
        super().__init__("WORLD")


class Centering(CoordinateSystem):
    """中心化坐标系。原点 = 目标中心在 World 中的位置。

    通过 cor_type 区分 LCA/RCA，方便根据输入选择：
        Centering(CoronaryType.LCA, translation)
        Centering(CoronaryType.RCA, translation)
    """
    translation: np.ndarray  # (3,)
    cor_type: CoronaryType

    def __init__(self, cor_type: CoronaryType, translation: np.ndarray):
        assert translation.shape == (3,), f"Expected (3,), got {translation.shape}"
        super().__init__(f"{cor_type.value}_CENTERING")
        self.cor_type = cor_type
        self.translation = translation.copy()

    @staticmethod
    def from_cor_type(cor_type: CoronaryType, translation: np.ndarray) -> "Centering":
        """根据 CoronaryType 创建对应的 Centering。"""
        return Centering(cor_type, translation)


# ─── TypeVar ───

CS = TypeVar("CS", World, Centering)
"""泛型约束：MedImage[World | Centering]"""


# ─── 医学图像类 ───

class MedImage(Generic[CS]):
    """泛型医学图像，类型参数 CS 标记当前坐标系。

    用法:
        img_w: MedImage[World] = MedImage(data, affine, World())
        img_c: MedImage[Centering] = img_w.to_coord(Centering(cor_type, center))

    裁剪不改变坐标系:
        cropped: MedImage[World] = img_w.crop(x0, x1, y0, y1, z0, z1)
    """
    data: torch.Tensor
    affine: np.ndarray          # voxel → coord 的完整 4×4 affine
    coord: CS                   # 当前坐标系实例（携带 translation）

    def __init__(self, data: torch.Tensor, affine: np.ndarray, coord: CS):
        self.data = data
        self.affine = np.asarray(affine, dtype=np.float64)
        self.coord = coord

    def to_coord(self, target_cs: CoordinateSystem) -> MedImage:
        """变换到目标坐标系（纯平移，affine 的 B 部分不变）。"""
        T = self.coord.transform_to(target_cs)
        return MedImage(
            data=self.data,
            affine=T @ self.affine,
            coord=target_cs,          # type: ignore
        )

    def crop(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> MedImage[CS]:
        """裁剪 — 数据范围改变，坐标系和 affine 不变。"""
        return MedImage(
            data=self.data[..., x0:x1, y0:y1, z0:z1],
            affine=self.affine,
            coord=self.coord,
        )