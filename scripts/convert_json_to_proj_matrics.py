#! /usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "nibabel"]
# ///


# C-arm 投影矩阵生成 (Projection Matrix Generator)
# =================================================
#
# 基于 C-arm 几何参数和旋转角度 (alpha, beta) 生成 3D→2D 投影矩阵。
# 坐标系约定：
#   - 世界坐标系: RAS 坐标系， X指向患者右方 , Y 指向患者前方 , Z 指向患者头部
#   - 相机坐标系: 由 camera_convention 参数决定, 默认 "pytorch3d" (= DiffDRR/PyTorch3D:
#     +X 左, +Y 上, +Z 前), 与 DRR 渲染一致, 可复用渲染器的 R/T。
#     投影数学为 p = K @ View @ P_w (见下), 深度 = P·forward, 正值表示点在相机前方;
#     forward 方向由约定决定 (pytorch3d: +Z, opengl: -Z)。
#     外参 reorient 由两个映射直接构造: _ORIENTATION_POSE (物理方向在世界坐标中的指向)
#     与 CONVENTION_AXES (物理方向在目标相机坐标中对应的相机轴), 不再经过 z-up 中间帧。
#   - 图像坐标系: DICOM/OpenCV 约定, 原点在左上角, u 向右, v 向下
#
# 投影公式：
#   p = K @ View @ P_w                 (齐次坐标, 4×4)
#   u = p[0] / p[2],  v = p[1] / p[2]   (透视除法)
#   (相机 X/Y/Z 的符号约定已编码进内参 K, 无需再额外翻转)
#
# 内参矩阵 K (4×4, 与 camera_convention 相关):
#   设 (right, down, forward) 为该约定的图像轴(目标相机坐标), 则
#   K = [[ fx*right + cx*forward ],
#        [ fy*down  + cy*forward ],
#        [ forward ],
#        [ 0 0 0 1 ]]
#   fx = sdd/delx, fy = sdd/dely, cx = width/2, cy = height/2
#   图像轴由 R_conv 推导: right=R[0,:], down=R[1,:], forward=R[2,:]
#   (R_conv 列 = 目标约定相机轴在 OpenCV 基下, 故目标相机坐标下的图像轴为 R 的行).
#   对标准约定与 OpenCV 约定 (R=I) 均适用.
#
# 外参矩阵 View (4×4, world-to-camera)：
#   view = rot_zxy(alpha, beta, 0) @ translate @ reorient
#   m_w2c = inv(m_c2w)
#   - reorient 将世界坐标映射到目标约定相机坐标系, 其列 = 相机轴 (X,Y,Z) 在世界坐标中的方向
#     (AP 时相机前向指向 -Y, 即从患者前方 A 指向后方 P; PA 时指向 +Y)
#   - translate 将相机置于 (0, sod, 0) 位置
#   - rot 为欧拉角旋转矩阵, 旋转轴顺序为 ZXY, 旋转角度为 (alpha, beta, 0)
#
# 投影矩阵 P (3×4)：
#  P = K @ View[:3, :] = [k | 0] @ [R, T]
#
# 深度值 (depth)：
#   depth = p[:, 2] = P·forward
#   正值表示点在相机前方, 单位为世界坐标单位 (mm)


import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from cyclopts import run
import pytest


def _axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """Rotation matrix (4x4) about a single axis for each given Euler angle.

    Args:
        axis: Axis label "X", "Y" or "Z".
        angle: Euler angles in radians, shape (..., 1).

    Returns:
        Rotation matrices of shape (..., 4, 4).
    """
    # Remove trailing dimension from np.split output (..., 1) -> (...,)
    angle = angle[..., 0]

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    # Build result with shape (..., 4, 4) to support both scalar and batched inputs
    result = np.zeros(angle.shape + (4, 4))
    result[..., 0, 0] = R_flat[0]
    result[..., 0, 1] = R_flat[1]
    result[..., 0, 2] = R_flat[2]
    result[..., 1, 0] = R_flat[3]
    result[..., 1, 1] = R_flat[4]
    result[..., 1, 2] = R_flat[5]
    result[..., 2, 0] = R_flat[6]
    result[..., 2, 1] = R_flat[7]
    result[..., 2, 2] = R_flat[8]
    result[..., 3, 3] = 1.0
    return result

def euler_angles_to_matrix(convention: str, euler_angles: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as array of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", "Z"}.

    Returns:
        Rotation matrices as array of shape (..., 4, 4).
    """
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")

    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")

    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")

    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, np.split(euler_angles, 3, axis=-1))
    ]
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])

# CameraOrientation: physical directions (right/front/up) as signed world/camera axes.
# "front" is the X-ray beam direction.

OrientPose = Literal["AP", "PA"]
CameraType = Literal["pytorch3d", "opencv", "diffdrr", "nerfstudio", "gs", "opengl", "nerf", "colmap", "zup"]
CameraDirection = Literal["right", "front", "up"]
CoordinateDirection = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

Vec3 = NDArray[np.float64]

class CameraOrientation(TypedDict):
    right: CoordinateDirection
    front: CoordinateDirection
    up: CoordinateDirection


# _ORIENTATION_POSE: physical directions (right/front/up) of the C-arm in WORLD (RAS)
# coordinates. Camera-agnostic: fixes where the camera looks, independent of how the
# camera axes are named.
# AP: source in front of the patient (+Y), beam anterior -> posterior => front = -Y
# PA: source behind the patient (-Y),  beam posterior -> anterior   => front = +Y
_ORIENTATION_POSE: dict[OrientPose, CameraOrientation] = {
    # orientation -> {right, front, up} as world-axis specs ("+X", "-Z", ...)
    "AP": {"right": "-X", "front": "-Y", "up": "+Z"},
    "PA": {"right": "+X", "front": "+Y", "up": "+Z"},
}

# CONVENTION_AXES: physical directions (right/front/up) expressed in the TARGET camera
# frame, i.e. which camera axis (+/-X/Y/Z) each physical direction points along.
#   - PyTorch3D / DiffDRR : +X 左, +Y 上, +Z 前   => right=-X, front=+Z, up=+Y
#   - OpenGL / NeRF       : +X 右, +Y 上, -Z 前   => right=+X, front=-Z, up=+Y
#   - COLMAP / OpenCV / GS: +X 右, +Y 下, +Z 前   => right=+X, front=+Z, up=-Y
#   - zup                 : x右, y前, z上         => right=+X, front=+Y, up=+Z
CONVENTION_AXES: dict[CameraType, CameraOrientation] = {
    "pytorch3d":    {"right": "-X", "front": "+Z", "up": "+Y"},
    "diffdrr":      {"right": "-X", "front": "+Z", "up": "+Y"},
    "opengl":       {"right": "+X", "front": "-Z", "up": "+Y"},
    "nerf":         {"right": "+X", "front": "-Z", "up": "+Y"},
    "colmap":       {"right": "+X", "front": "+Z", "up": "-Y"},
    "opencv":       {"right": "+X", "front": "+Z", "up": "-Y"},
    "gs":           {"right": "+X", "front": "+Z", "up": "-Y"},
    "zup":          {"right": "+X", "front": "+Y", "up": "+Z"},
}

# CONVENTION_ROTATIONS: hardcoded ground-truth R_conv (columns = target-convention
# camera axes X/Y/Z in the OpenCV basis), kept for internal testing of CONVENTION_AXES.
CONVENTION_ROTATIONS: dict[CameraType, np.ndarray] = {
    "pytorch3d": np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ]),
    "diffdrr": np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ]),
    "opengl": np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]),
    "nerf": np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]),
    "colmap": np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]),
    "opencv": np.eye(3),  # OpenCV 相机: x右, y下, z前 (R_conv = I)
    "gs": np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]),
    "zup": np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]),
}



def _axis_vec(axis_spec: str) -> Vec3:
    """Convert a world-axis spec such as "+X" or "-Z" into a unit vector."""
    sign = -1.0 if axis_spec.startswith("-") else 1.0
    letter = axis_spec.lstrip("+-")
    if letter not in "XYZ":
        raise ValueError(f"Invalid world axis spec: {axis_spec!r}")
    vec = np.zeros(3)
    vec["XYZ".index(letter)] = sign
    return vec


# OpenCV 基下的物理方向单位向量 (x=right, y=down, z=forward)
_OPENCV_BASIS: dict[str, Vec3] = {
    "right": np.array([1.0, 0.0, 0.0]),
    "front": np.array([0.0, 0.0, 1.0]),
    "up": np.array([0.0, -1.0, 0.0]),
}


def _resolve_convention_axes(camera_convention: CameraType | CameraOrientation) -> CameraOrientation:
    """Resolve a camera-convention spec into a {right/front/up: signed X/Y/Z} mapping."""
    if isinstance(camera_convention, str):
        name = camera_convention.lower()
        if name not in CONVENTION_AXES:
            raise ValueError(
                f"Unknown camera convention {camera_convention!r}; choose from "
                f"{sorted(CONVENTION_AXES)} or pass an explicit axis mapping."
            )
        return CONVENTION_AXES[name]
    if isinstance(camera_convention, dict):
        if set(camera_convention) != {"right", "front", "up"}:
            raise ValueError("camera_convention dict must have exactly keys right/front/up")
        return camera_convention
    raise TypeError(
        "camera_convention must be a str (convention name) or a dict {right/front/up: ...}"
    )


def _camera_axes_in_physical(axes: CameraOrientation) -> list[tuple[float, str]]:
    """Invert a {right/front/up: signed X/Y/Z} mapping into per-camera-axis
    (sign, physical-direction) pairs, e.g. {"right": "-X", "front": "+Z", "up": "+Y"}
    -> [(-1, "right"), (+1, "up"), (+1, "front")] for camera axes X, Y, Z.
    """
    out = []
    for axis in "XYZ":
        for phys in ("right", "front", "up"):
            expr = axes[phys]
            if expr.lstrip("+-") == axis:
                sign = -1.0 if expr.startswith("-") else 1.0
                out.append((sign, phys))
                break
        else:
            raise ValueError(f"camera_convention axes mapping missing axis {axis!r}")
    return out


def _convention_rotation_from_axes(axes: CameraOrientation) -> np.ndarray:
    """Build R_conv (3x3) from a {right/front/up: signed X/Y/Z} mapping.

    R_conv columns are the target-convention camera axes (X, Y, Z) expressed in
    the OpenCV basis (x=right, y=down, z=forward).
    """
    cols = [sign * _OPENCV_BASIS[phys] for sign, phys in _camera_axes_in_physical(axes)]
    rot3 = np.stack(cols, axis=1)
    if not np.isclose(np.linalg.det(rot3), 1.0):
        raise ValueError(
            "camera_convention axis mapping must be a proper rotation (det = +1)"
        )
    return rot3


def _resolve_convention_rotation(camera_convention: CameraType | CameraOrientation) -> np.ndarray:
    """Resolve a camera-convention spec into the convention rotation R_conv (4x4).

    R_conv columns are the target-convention camera axes (X, Y, Z) expressed in the
    OpenCV basis (x=right, y=down, z=forward). Accepts a convention name, or an
    explicit dict mapping each physical direction (right/front/up) to a signed
    camera axis (+/-X/Y/Z).
    """
    rot3 = _convention_rotation_from_axes(_resolve_convention_axes(camera_convention))
    rot4 = np.eye(4)
    rot4[:3, :3] = rot3
    return rot4


def _resolve_image_axes(camera_convention: CameraType | CameraOrientation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive the convention's image axes (right, down, forward) in the target
    camera frame from R_conv.

    In the OpenCV basis the physical image axes are right=+x, down=+y, forward=+z
    (the OpenCV convention itself has R_conv = I); R_conv maps target coords ->
    OpenCV basis, so the image axes in the TARGET frame are the rows of R_conv:
      right   = R_conv[0, :]
      down    = R_conv[1, :]
      forward = R_conv[2, :]
    Valid for any convention, including OpenCV (R_conv = I) and +Z-backward ones
    (e.g. OpenGL).
    """
    rot = _resolve_convention_rotation(camera_convention)[:3, :3]
    return rot[0, :], rot[1, :], rot[2, :]


def _reorientation_from_pose(pose: CameraOrientation, axes: CameraOrientation) -> np.ndarray:
    """Build the world->camera reorientation matrix directly from two axis mappings.

    pose: {right/front/up: signed world axis} — physical directions in world (RAS) coords.
    axes: {right/front/up: signed camera axis} — physical directions in the target camera frame.
    Returns a 4x4 matrix whose COLUMNS are the target camera axes (X, Y, Z) in world coords.
    """
    cols = []
    for sign_conv, phys in _camera_axes_in_physical(axes):
        world_expr = pose[phys]
        sign_orient = -1.0 if world_expr.startswith("-") else 1.0
        cols.append(sign_conv * sign_orient * _axis_vec(world_expr.lstrip("+-")))
    reorient = np.eye(4)
    reorient[:3, :3] = np.stack(cols, axis=1)  # columns = camera axes in world coords
    return reorient


def get_reorientation(
    orientation_type: str = "AP",
    camera_convention: CameraType | CameraOrientation = "pytorch3d",
) -> np.ndarray:
    """
    Get the 4x4 reorientation matrix (world -> camera axes) for a C-arm orientation
    (AP/PA) and a camera coordinate convention.

    Built directly from two axis mappings (no intermediate z-up frame):
      1. _ORIENTATION_POSE: physical directions (right/front/up) in world (RAS) coords
         (AP -> front=-Y, PA -> front=+Y).
      2. CONVENTION_AXES: physical directions in the target camera frame (which
         camera axis +/-X/Y/Z each points along).
    The columns of the result are the target-convention camera axes (X, Y, Z) in
    world coordinates.

    camera_convention may be:
      - a name: "pytorch3d" | "diffdrr" | "opengl" | "nerf" | "colmap" | "opencv" | "gs" | "zup"
      - an explicit dict {"right": ..., "front": ..., "up": ...} mapping each physical
        direction to a signed camera axis (+/-X/Y/Z), e.g.
        {"right": "+X", "front": "-Z", "up": "+Y"} (OpenGL).

    The default "pytorch3d" matches DiffDRR / PyTorch3D (+X left, +Y up, +Z forward),
    so the produced R/T are identical to the DRR renderer's (reuse R/T).

    Args:
        orientation_type: "AP" (Anterior-Posterior) or "PA" (Posterior-Anterior).
        camera_convention: camera coordinate convention, default "pytorch3d".

    Returns:
        reorient: (4, 4) reorientation matrix.
    """
    if orientation_type not in _ORIENTATION_POSE:
        raise ValueError(
            f"orientation_type must be either AP or PA, but got {orientation_type}"
        )
    pose = _ORIENTATION_POSE[orientation_type]
    axes = _resolve_convention_axes(camera_convention)
    return _reorientation_from_pose(pose, axes)


def get_view_matrix(
    alpha: float,
    beta: float,
    sod: float,
    orientation_type: str = "AP",
    camera_convention: CameraType | CameraOrientation = "pytorch3d",
) -> np.ndarray:
    """
    Build the world-to-camera view matrix (m_c2w = rot @ translate @ reorient, then
    invert). The camera axes (extrinsics) follow the given camera convention; the
    default "pytorch3d" (= DiffDRR/PyTorch3D) matches the DRR renderer so R/T can be
    reused directly.
    Args:
        alpha: float: rotation angle around Z-axis in radians
        beta: float: rotation angle around X-axis in radians
        sod: float: source to object distance in world coordinate system
        orientation_type: str: "AP" or "PA" C-arm orientation, default "AP"
        camera_convention: camera coordinate convention (see get_reorientation),
            default "pytorch3d"
    Returns:
        View Matrix: (4, 4) array [R, T; 0, 1] in world coordinate system
    """
    reorient = get_reorientation(orientation_type, camera_convention)
    translate = np.eye(4)
    sod_sign = 1.0 if orientation_type == "AP" else -1.0
    translate[:3, 3] = np.array([0.0, sod_sign * sod, 0.0])
    rot = euler_angles_to_matrix("ZXY", np.array([alpha, beta, 0.0]))
    m_c2w = rot @ translate @ reorient  # [R_c2w, T_c2w; 0, 1]
    m_w2c = np.linalg.inv(m_c2w)        # [R_c2w^T, -R_c2w^T*T_c2w; 0, 1]
    return m_w2c


def get_view_matrixs(
    alphas: np.ndarray,
    betas: np.ndarray,
    sod: float,
    orientation_type: str = "AP",
    camera_convention: CameraType | CameraOrientation = "pytorch3d",
) -> np.ndarray:
    """
    Get the view matrices for a batch of camera poses.

    Args:
        alphas: (N,) array of rotation angles around Z-axis in radians
        betas: (N,) array of rotation angles around X-axis in radians
        sod: float: source to object distance in world coordinate system
        orientation_type: str: "AP" or "PA" C-arm orientation, default "AP"
        camera_convention: camera coordinate convention (see get_reorientation),
            default "pytorch3d"

    Returns:
        view_matrices: (N, 4, 4) array of view matrices [R, T; 0, 1] in world coordinate system
    """
    reorient = get_reorientation(orientation_type, camera_convention)
    translate = np.eye(4)
    sod_sign = 1.0 if orientation_type == "AP" else -1.0
    translate[:3, 3] = np.array([0.0, sod_sign * sod, 0.0])
    rot = euler_angles_to_matrix("ZXY", np.stack([alphas, betas, np.zeros_like(alphas)], axis=-1))
    m_c2w = rot @ translate @ reorient  # [R_c2w, T_c2w; 0, 1]
    m_w2c = np.linalg.inv(m_c2w)        # [R_c2w^T, -R_c2w^T*T_c2w; 0, 1]
    return m_w2c


def get_intrinsic_matrix(
    sdd: float,
    width: int,
    height: int,
    delx: float,
    dely: float,
    x0: float = 0.0,
    y0: float = 0.0,
    camera_convention: CameraType | CameraOrientation = "pytorch3d",
) -> np.ndarray:
    """
    Build the convention-aware intrinsic matrix K.

    K encodes the camera convention's image axes so that `u = p0/p2, v = p1/p2`
    directly yields DICOM/OpenCV pixels (u right, v down, top-left origin) with
    p2 = depth > 0 for points in front of the camera:
      K = [[ fx*right + cx*forward ],
           [ fy*down  + cy*forward ],
           [ forward ],
           [ 0 0 0 1 ]]
    where (right, down, forward) are the convention's image axes in the target
    camera frame (see _resolve_image_axes): u = cx + fx*(P.right)/(P.forward),
    v = cy + fy*(P.down)/(P.forward), depth = P.forward.

    Args:
        sdd: float: source to detector distance
        width: int: detector width in pixels
        height: int: detector height in pixels
        delx: float: pixel size in x direction
        dely: float: pixel size in y direction
        x0: float: x coordinate of the principal point in pixels
        y0: float: y coordinate of the principal point in pixels
        camera_convention: camera coordinate convention (see get_reorientation),
            default "pytorch3d"
    Returns:
        intrinsic_matrix: (4, 4) array of intrinsic matrix K (camera coordinates)
    """
    fx = sdd / delx
    fy = sdd / dely
    cx = width / 2.0 + x0
    cy = height / 2.0 + y0
    right, down, forward = _resolve_image_axes(camera_convention)
    # K rows = [fx*right + cx*forward, fy*down + cy*forward, forward]
    row_u = fx * right + cx * forward
    row_v = fy * down + cy * forward
    return np.array([
        [row_u[0], row_u[1], row_u[2], 0.0],
        [row_v[0], row_v[1], row_v[2], 0.0],
        [forward[0], forward[1], forward[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


@dataclass
class Projector:
    sdd: float
    sod: float
    width: int
    height: int
    delx: float
    dely: float
    x0: float = 0.0
    y0: float = 0.0
    orientation_type: str = "AP"
    camera_convention: CameraType | CameraOrientation = "pytorch3d"

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Convention-aware intrinsic matrix K, recomputed when camera_convention changes."""
        return get_intrinsic_matrix(
            self.sdd, self.width, self.height, self.delx, self.dely,
            self.x0, self.y0, self.camera_convention,
        )   # (4, 4)

    def project(
        self,
        alpha: float,
        beta: float,
        points_3d: np.ndarray,
        return_depth: bool = False
    ) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
        """
        Args:
            alpha: float: rotation angle around Z-axis in radians
            beta: float: rotation angle around X-axis in radians
            points_3d: (N, 3) array of 3D points in world coordinate system
            return_depth: bool: if True, also return depth (distance from camera along view direction)
        Returns:
            (u, v) coordinates of projected points on detector plane in pixels at DICOM/openCV convention
            where origin is at the top-left corner of the image, u is the horizontal axis and v is the vertical axis.
            If return_depth=True, returns ((N, 2), (N,)) tuple of (uv, depth).
        """
        # Get projection matrix View in world coordinate system
        view_matrix = get_view_matrix(alpha, beta, self.sod, self.orientation_type, self.camera_convention)    # (4, 4)
        # Project 3D points to 2D points in homogeneous coordinates
        point_3d_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)  # (N, 4)
        point_2d_h = (self.intrinsic_matrix @ view_matrix @ point_3d_h.T).T  # (N, 4)  (x, y, z, w=1)
        # depth = point_2d_h[:, 2]; positive for points in front of the camera
        # (the sign of Z is encoded in the convention-aware intrinsic matrix)
        depth = point_2d_h[:, 2]
        # Convert to pixel coordinates, u = x / z, v = y / z.
        # The camera convention's X/Y/Z sign is baked into K, so no extra flip is
        # needed to reach DICOM/OpenCV pixels (u right, v down, top-left origin).
        u = point_2d_h[:, 0] / depth
        v = point_2d_h[:, 1] / depth

        uv = np.stack([u, v], axis=1)  # (N, 2)
        if return_depth:
            return uv, depth
        return uv

    def project_to_png(
        self,
        alpha: float,
        beta: float,
        points_3d: np.ndarray,
        png_path: Path
    ) -> None:
        """
        Project 3D points to 2D points in pixel coordinates and save to a PNG file,
        with color encoding the depth (distance from camera).

        Args:
            alpha: float: rotation angle around Z-axis in radians
            beta: float: rotation angle around X-axis in radians
            points_3d: (N, 3) array of 3D points in world coordinate system
            png_path: Path to save the projected points as a PNG file
        """
        import matplotlib.pyplot as plt

        uv, depth = self.project(alpha, beta, points_3d, return_depth=True)  # (N, 2), (N,)

        # Sort by depth descending (far to near), so near points draw on top
        sort_idx = np.argsort(-depth)
        uv_sorted = uv[sort_idx]
        depth_sorted = depth[sort_idx]

        fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=100)
        scatter = ax.scatter(
            uv_sorted[:, 0], uv_sorted[:, 1], c=depth_sorted, cmap='plasma', s=1,
            vmin=500, vmax=1000
        )
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()  # Invert y-axis to match DICOM/OpenCV convention
        ax.set_title(f"alpha={np.rad2deg(alpha):.2f}°, beta={np.rad2deg(beta):.2f}°")
        ax.set_aspect('equal')

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Depth (mm)')

        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def from_json_data(data: dict[str, Any]) -> "Projector":
        c_arm_geom = data.get("c_arm_geometry", {})
        orientation_type = c_arm_geom.get("orientation_type", None)
        if orientation_type is None:
            import warnings
            warnings.warn(
                "orientation_type not found in JSON, defaulting to 'AP'. "
                "Please ensure your JSON contains the correct orientation_type."
            )   

        return Projector(
            sdd=c_arm_geom["sdd"],
            sod=c_arm_geom["sod"],
            width=c_arm_geom["width"],
            height=c_arm_geom["height"],
            delx=c_arm_geom["delx"],
            dely=c_arm_geom["dely"],
            x0=c_arm_geom.get("x0", 0.0),
            y0=c_arm_geom.get("y0", 0.0),
            orientation_type=orientation_type or "AP",
            camera_convention="diffdrr"     # data is generated by DiffDRR, so use its convention as default
        )


    def get_projection_matrix(
        self,
        alpha: float,
        beta: float
    ) -> np.ndarray:
        """
        Get the 3x4 projection matrix P = K @ View (world -> pixels).

        Args:
            alpha: float: rotation angle around Z-axis in radians
            beta: float: rotation angle around X-axis in radians
        """
        M = self.intrinsic_matrix @ get_view_matrix(alpha, beta, self.sod, self.orientation_type, self.camera_convention)  # (4, 4)
        return M[:3, :]  # (3, 4)

    def get_projection_matrices(
        self,
        alphas: np.ndarray,
        betas: np.ndarray
    ) -> np.ndarray:
        """
        Get the (N, 3, 4) projection matrices P = K @ View (world -> pixels).

        P is convention-INVARIANT: the physical camera/detector is the same for
        every convention, so the 2D coordinates and depths are identical; only
        the extrinsic representation (R/T) differs (see get_view_matrices).

        Args:
            alphas: (N,) array of rotation angles around Z-axis in radians
            betas: (N,) array of rotation angles around X-axis in radians
        """
        view_matrices = get_view_matrixs(alphas, betas, self.sod, self.orientation_type, self.camera_convention)  # (N, 4, 4)
        M = self.intrinsic_matrix @ view_matrices  # (N, 4, 4)
        return M[:, :3, :]  # (N, 3, 4)

    def get_view_matrices(
        self,
        alphas: np.ndarray,
        betas: np.ndarray
    ) -> np.ndarray:
        """
        Get the (N, 4, 4) world-to-camera view matrices [R_w2c, T_w2c; 0, 1].

        The camera convention only changes this extrinsic representation (R/T),
        which is what downstream libraries (PyTorch3D, COLMAP, nerfstudio,
        gsplat, ...) consume; the projection matrix P = K @ View is invariant.

        Args:
            alphas: (N,) array of rotation angles around Z-axis in radians
            betas: (N,) array of rotation angles around X-axis in radians
        """
        return get_view_matrixs(alphas, betas, self.sod, self.orientation_type, self.camera_convention)  # (N, 4, 4)


PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def generate_static_projection_assets(
    labe_nii_path: Path = PROJECT_ROOT / "test/test_data/rbf_reader_flow_contrast_LCA/coronary_label.nii.gz",
    json_path: Path = PROJECT_ROOT / "test/test_data/rbf_reader_flow_contrast_LCA/rotate_dsa.json",
    out_dir: Path = PROJECT_ROOT / "test/test_data/rbf_reader_flow_contrast_LCA/_test_assets",
    down_sample_stride: int = 10,
    camera_convention: CameraType = "diffdrr",
) -> None:
    """Project the static 3D label with the reference convention and save the
    per-frame 2D projections as reusable test assets.

    The synthetic DSA data is dynamic (the heart moves between frames), so the
    rendered labels move; the static-label projection is the geometry ground
    truth and is reused by test_projector for verification.

    Saves:
      - static_projection.npy: (N_frames, N_points, 3) [u, v, depth]
      - points_world.npy:      (N_points, 3) world coordinates of the label points
      - meta.json:             generation parameters (convention, frames, ...)
    """
    import nibabel as nib

    with open(json_path, "r") as f:
        json_data = json.load(f)

    label_nii = nib.load(labe_nii_path)
    assert isinstance(label_nii, nib.Nifti1Image), "label_nii must be a Nifti1Image"
    data = label_nii.get_fdata()
    affine = np.array(json_data["lca_centering_affine"])
    pts = np.argwhere(data > 0)[::down_sample_stride]  # (M, 3)
    points_world = pts @ affine[:3, :3].T + affine[:3, 3]  # (M, 3)

    projector = Projector.from_json_data(json_data)
    projector.camera_convention = camera_convention

    frames = json_data["frames"]
    alphas = np.deg2rad([f["alpha_degree"] for f in frames])
    betas = np.deg2rad([f["beta_degree"] for f in frames])
    P = projector.get_projection_matrices(alphas, betas)  # (N, 3, 4)
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)  # (M, 4)
    proj = P @ pts_h.T  # (N, 3, M)
    depth = proj[:, 2, :]  # (N, M)
    u = proj[:, 0, :] / depth
    v = proj[:, 1, :] / depth
    static_projection = np.stack([u, v, depth], axis=-1)  # (N, M, 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "static_projection.npy", static_projection)
    np.save(out_dir / "points_world.npy", points_world)
    meta = {
        "camera_convention": camera_convention,
        "orientation_type": projector.orientation_type,
        "down_sample_stride": down_sample_stride,
        "n_points": len(points_world),
        "n_frames": len(frames),
        "frames": [
            {"alpha_degree": f["alpha_degree"], "beta_degree": f["beta_degree"]}
            for f in frames
        ],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved static projection assets to {out_dir}")


def test_convention_rotations() -> None:
    """Verify R_conv computed from CONVENTION_AXES matches the hardcoded ground truth."""
    for name, gt in CONVENTION_ROTATIONS.items():
        computed = _resolve_convention_rotation(name)[:3, :3]
        assert np.allclose(computed, gt), (
            f"convention {name!r} mismatch:\n{computed}\nvs\n{gt}"
        )


def test_projector(
    assets_dir: Path = PROJECT_ROOT / "test/test_data/rbf_reader_flow_contrast_LCA/_test_assets",
    json_path: Path = PROJECT_ROOT / "test/test_data/rbf_reader_flow_contrast_LCA/rotate_dsa.json",
    out_dir: Path = Path(r"temp"),
    n_max: int = 30,
    frame_stride: int = 2,
    save_visualization: bool | None = None,
) -> None:
    """
    Verify projection invariance and extrinsic representation per convention.

    The physical setup (world, camera pose, detector) is the same for every
    camera convention, so the projection matrix P = K @ View and the resulting
    2D coordinates / depths are convention-INVARIANT. What differs between
    conventions is only the extrinsic representation (R/T), which is what
    downstream libraries (PyTorch3D, COLMAP, nerfstudio, gsplat, ...) consume.

    Checks:
      1. Invariance: every convention reproduces the reference projections
         (generated by generate_static_projection_assets with "diffdrr") to
         sub-pixel accuracy.
      2. Extrinsics differ: "diffdrr" / "pytorch3d" (identical convention) share
         the same View; every other convention has a different View.
      3. Extrinsics match the convention: the rows of the View rotation are the
         camera axes in world coordinates, as derived from CONVENTION_AXES.

    Overlay PNGs (current convention in red, reference in green) are saved to
    `out_dir/<convention>/` only when `save_visualization=True` or the env var
    `PROJECTOR_SAVE_PNG` is set to any of "1/true/yes/on".

    Test by: `uv run --no-project pytest scripts/convert_json_to_proj_matrics.py`
    """
    import os
    import matplotlib.pyplot as plt

    if save_visualization is None:
        save_visualization = os.environ.get("PROJECTOR_SAVE_PNG", "").strip().lower() in ("1", "true", "yes", "on")

    static_projection = np.load(assets_dir / "static_projection.npy")  # (N, M, 3) [u, v, depth]
    points_world = np.load(assets_dir / "points_world.npy")  # (M, 3)
    with open(assets_dir / "meta.json", "r") as f:
        meta = json.load(f)
    ref_uv = static_projection[..., :2]
    ref_valid = static_projection[..., 2] > 0
    print(f"Loaded {len(points_world)} reference points, {static_projection.shape[0]} frames from {assets_dir}")

    with open(json_path, "r") as f:
        json_data = json.load(f)
    projector = Projector.from_json_data(json_data)
    frame_indices = list(range(0, static_projection.shape[0], frame_stride))[:n_max]

    mean_dists: dict[str, float] = {}
    views: dict[str, np.ndarray] = {}
    for convention in CONVENTION_AXES:
        projector.camera_convention = convention
        dists = []
        for fi in frame_indices:
            frame = json_data["frames"][fi]
            uv, depth = projector.project(
                np.deg2rad(frame["alpha_degree"]), np.deg2rad(frame["beta_degree"]),
                points_world, return_depth=True,
            )
            valid = ref_valid[fi] & (depth > 0)  # both in front of the camera
            if valid.sum() == 0:
                continue
            d = np.linalg.norm(uv[valid] - ref_uv[fi][valid], axis=1)
            dists.append(d)

            if save_visualization:
                fig, ax = plt.subplots()
                ax.scatter(ref_uv[fi][valid, 0], ref_uv[fi][valid, 1], s=1, c="green", label="reference")
                ax.scatter(uv[valid, 0], uv[valid, 1], s=1, c="red", label=convention)
                ax.set_title(
                    f"{convention} frame {fi} "
                    f"(alpha={frame['alpha_degree']}°, beta={frame['beta_degree']}°)"
                )
                ax.legend(markerscale=5)
                save_dir = out_dir / convention
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / f"frame_{fi:03d}.png", dpi=100, bbox_inches="tight")
                plt.close(fig)

        mean_dists[convention] = float(np.concatenate(dists).mean()) if dists else float("inf")
        alphas = np.deg2rad([json_data["frames"][fi]["alpha_degree"] for fi in frame_indices])
        betas = np.deg2rad([json_data["frames"][fi]["beta_degree"] for fi in frame_indices])
        views[convention] = get_view_matrixs(
            alphas, betas, projector.sod, projector.orientation_type, convention,
        )
        print(f"  {convention:10s}: mean_dist={mean_dists[convention]:8.3f} px")

    # 1) Projection invariance: P = K @ View is convention-independent.
    for conv, d in mean_dists.items():
        assert d < 1.0, f"{conv} mean dist {d:.3f} px too large (projection must be invariant)"

    # 2) Extrinsics differ between conventions (only the R/T representation changes).
    v_ref = views["diffdrr"]
    assert np.allclose(views["pytorch3d"], v_ref), "pytorch3d and diffdrr must share the same View"
    for conv, v in views.items():
        if conv not in ("diffdrr", "pytorch3d"):
            assert not np.allclose(v, v_ref, atol=1e-6), f"{conv} View unexpectedly identical to diffdrr"

    # 3) Extrinsics match the convention: rows of the View rotation are the camera
    #    axes in world coordinates, as derived from CONVENTION_AXES.
    for conv in CONVENTION_AXES:
        pose = _ORIENTATION_POSE[projector.orientation_type]  # type: ignore[index]
        axes = _resolve_convention_axes(conv)
        for fi in frame_indices[:1]:
            frame = json_data["frames"][fi]
            rot = euler_angles_to_matrix("ZXY", np.array([np.deg2rad(frame["alpha_degree"]), np.deg2rad(frame["beta_degree"]), 0.0]))[:3, :3]
            R_w2c = views[conv][frame_indices.index(fi)][:3, :3]
            for axis_idx, axis_letter in enumerate("XYZ"):
                # camera axis `axis_letter` in world coords, from the convention mapping
                cam_axis = None
                for phys in ("right", "front", "up"):
                    expr = axes[phys]
                    if expr.lstrip("+-") == axis_letter:
                        sign = -1.0 if expr.startswith("-") else 1.0
                        world_expr = pose[phys]
                        sign_w = -1.0 if world_expr.startswith("-") else 1.0
                        cam_axis = sign * sign_w * _axis_vec(world_expr.lstrip("+-"))
                        break
                assert cam_axis is not None
                # R_w2c row = camera axis in world coords, rotated by the C-arm pose
                assert np.allclose(R_w2c[axis_idx], rot @ cam_axis, atol=1e-6), (
                    f"{conv} View row {axis_idx} does not match camera axis {axis_letter}"
                )


def main(
    json_path: Path,
    output_path: Path|None = None,
    output_type: Literal["npy", "json"] = "npy",
    orientation: OrientPose|None = None,
    camera_convention: CameraType = "pytorch3d",
    save_extrinsics: bool = True,
):
    """
    Convert camera parameters and poses from a JSON file to projection matrices and save them to a .npy or .json file.

    The projection matrix P = K @ View is convention-INVARIANT (the physical
    camera/detector is the same for every convention); the camera convention only
    changes the extrinsic representation R_w2c / T_w2c. When save_extrinsics is
    True (default), both are saved:
      - npy  -> .npz with keys "projection_matrices" (N,3,4), "R_w2c" (N,3,3), "T_w2c" (N,3)
      - json -> dict with the same keys

    Args:
        json_path: Path to the JSON file containing camera parameters and poses.
        output_path: Path to save the projection matrices as .npy or .json file. If None, defaults to "projection_matrices.npy" in the same directory as the JSON file.
        output_type: Type of output file, either "npy" or "json". Default is "npy".
        orientation: C-arm orientation type, either "AP" or "PA". Default is None, which will read from the JSON file.
        camera_convention: Camera coordinate convention (see get_reorientation), default "pytorch3d".
        save_extrinsics: also save the world-to-camera extrinsics R_w2c / T_w2c, default True.
    """
    # Load projector from JSON
    print(f"Loading projector from JSON: {json_path}")
    
    with open(json_path, "r") as f:
        json_data = json.load(f)
    
    projector = Projector.from_json_data(json_data)
    if orientation is not None:
        print(f"Overriding orientation type from {projector.orientation_type} to {orientation}")
        projector.orientation_type = orientation
    
    print(f"Using camera convention: {camera_convention}")
    projector.camera_convention = camera_convention
    
    if output_path is None:
        output_path = json_path.parent / f"projection_matrices_{orientation}_{camera_convention}.{output_type}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = json_data.get("frames", [])
    alphas_degree = np.array([frame["alpha_degree"] for frame in frames])
    beta_degree = np.array([frame["beta_degree"] for frame in frames])
    alphas = np.deg2rad(alphas_degree)
    betas = np.deg2rad(beta_degree)

    projection_matrices = projector.get_projection_matrices(alphas, betas)  # (N, 3, 4)

    view_matrices = projector.get_view_matrices(alphas, betas)  # (N, 4, 4)
    R_w2c = view_matrices[:, :3, :3]  # (N, 3, 3)
    T_w2c = view_matrices[:, :3, 3]   # (N, 3)

    if output_path.suffix == ".npy":
        if save_extrinsics:
            output_path = output_path.with_suffix(".npz")
            np.savez(
                output_path,
                projection_matrices=projection_matrices,
                R_w2c=R_w2c,
                T_w2c=T_w2c,
            )
        else:
            # Save projection matrices to .npy file
            np.save(output_path, projection_matrices)
    elif output_path.suffix == ".json":
        with open(output_path, "w") as f:
            if save_extrinsics:
                json.dump({
                    "projection_matrices": projection_matrices.tolist(),
                    "R_w2c": R_w2c.tolist(),
                    "T_w2c": T_w2c.tolist(),
                }, f)
            else:
                json.dump(projection_matrices.tolist(), f)
    else:
        raise ValueError("Output path must have .npy or .json extension.")

if __name__ == "__main__":
    run(main)