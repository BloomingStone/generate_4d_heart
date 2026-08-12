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
#     内部投影数学为 OpenGL 风格 (相机看向 -Z), 外参 reorient 先按 z-up 相机
#     (x右, y前, z上, 前=射线方向) 推导, 再右乘约定旋转得到目标相机约定。
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
#   图像轴由 CONVENTION_ROTATIONS 推导: right=R[0,:], down=-R[2,:], forward=R[1,:]
#   (reorient = reorient_zup @ R_conv, 故目标相机坐标下的图像轴为 R 的行).
#   对标准约定与 z-up 约定 (R=I) 均适用.
#
# 外参矩阵 View (4×4, world-to-camera)：
#   view = rot_zxy(alpha, beta, 0) @ translate @ reorient
#   m_w2c = inv(m_c2w)
#   - reorient 将世界坐标映射到 OpenGL 相机坐标系，让相机前方朝向 -Y 方向（从患者前方A指向患者后方P）
#   - translate 将相机置于 (0, sod, 0) 位置
#   - rot 为欧拉角旋转矩阵, 旋转轴顺序为 ZXY, 旋转角度为 (alpha, beta, 0)
#
# 投影矩阵 P (3×4)：
#  P = K @ View[:3, :] = [k | 0] @ [R, T]
#
# 深度值 (depth)：
#   depth = p[:, 2] = -Z_camera
#   正值表示点在相机前方, 单位为世界坐标单位 (mm)


import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 4, 4).
    """

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
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as array of shape (..., 3, 3).
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
    # return functools.reduce(torch.matmul, matrices)
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])

# Physical camera pose in the canonical "z-up" camera frame (x=right, y=front, z=up),
# expressed in world (RAS) coordinates. "front" is the X-ray beam direction:
#   AP: source in front of the patient (+Y), beam anterior -> posterior => front = -Y
#   PA: source behind the patient (-Y),  beam posterior -> anterior   => front = +Y
# up = +Z (head) always; right = front x up (right-handed camera).
_ORIENTATION_POSE = {
    # orientation -> {right, front, up} as world-axis specs ("+X", "-Z", ...)
    "AP": {"right": "-X", "front": "-Y", "up": "+Z"},
    "PA": {"right": "+X", "front": "+Y", "up": "+Z"},
}

# Camera convention rotations: columns are the target-convention camera axes (X, Y, Z)
# expressed in the z-up camera basis (x=right, y=front, z=up). They are right-multiplied
# into the reorientation/projection matrix:
#   reorient = reorient_zup @ R_conv
# geometric meaning: first align the world to the z-up camera, then relabel its axes.
#   - PyTorch3D / DiffDRR : +X 左, +Y 上, +Z 前   (Z = forward)
#   - OpenGL / NeRF       : +X 右, +Y 上, -Z 前   (Z = -forward)
#   - COLMAP / OpenCV / GS: +X 右, +Y 下, +Z 前   (Y = -up, Z = forward)
CONVENTION_ROTATIONS = {
    "pytorch3d": np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ]),
    "diffdrr": np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ]),
    "opengl": np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]),
    "nerf": np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]),
    "colmap": np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]),
    "opencv": np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]),
    "gs": np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]),
    "zup": np.eye(3),  # z-up 相机: x右, y前, z上 (R_conv = I)
}

def _axis_vec(axis_spec: str) -> np.ndarray:
    """Convert a world-axis spec such as "+X" or "-Z" into a unit vector."""
    axis_spec = axis_spec.strip()
    sign = -1.0 if axis_spec.startswith("-") else 1.0
    letter = axis_spec.lstrip("+-")
    if letter not in "XYZ":
        raise ValueError(f"Invalid world axis spec: {axis_spec!r}")
    vec = np.zeros(3)
    vec["XYZ".index(letter)] = sign
    return vec


_ZUP_BASIS = {
    "right": np.array([1.0, 0.0, 0.0]),
    "front": np.array([0.0, 1.0, 0.0]),
    "up": np.array([0.0, 0.0, 1.0]),
}


def _resolve_convention_rotation(camera_convention: "str | dict[str, str]") -> np.ndarray:
    """Resolve a camera-convention spec into the convention rotation R_conv (4x4).

    R_conv columns are the target-convention camera axes (X, Y, Z) expressed in the
    z-up camera basis. Accepts a convention name, or an explicit dict mapping each
    camera axis to a signed direction among "right", "front", "up".
    """
    if isinstance(camera_convention, str):
        name = camera_convention.lower()
        if name not in CONVENTION_ROTATIONS:
            raise ValueError(
                f"Unknown camera convention {camera_convention!r}; choose from "
                f"{sorted(CONVENTION_ROTATIONS)} or pass an explicit axis mapping."
            )
        rot3 = CONVENTION_ROTATIONS[name]
    elif isinstance(camera_convention, dict):
        if set(camera_convention) != {"x", "y", "z"}:
            raise ValueError("camera_convention dict must have exactly keys x/y/z")
        cols = []
        for axis in ("x", "y", "z"):
            expr = camera_convention[axis]
            sign = -1.0 if expr.startswith("-") else 1.0
            base = expr.lstrip("+-")
            if base not in _ZUP_BASIS:
                raise ValueError(
                    f"Unknown direction {base!r}; expected right/front/up"
                )
            cols.append(sign * _ZUP_BASIS[base])
        rot3 = np.stack(cols, axis=1)
        if not np.isclose(np.linalg.det(rot3), 1.0):
            raise ValueError(
                "camera_convention axis mapping must be a proper rotation (det = +1)"
            )
    else:
        raise TypeError(
            "camera_convention must be a str (convention name) or a dict {x/y/z: ...}"
        )
    rot4 = np.eye(4)
    rot4[:3, :3] = rot3
    return rot4


def _resolve_image_axes(camera_convention: "str | dict[str, str]") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive the convention's image axes (right, down, forward) in the target
    camera frame, from its rotation matrix R_conv.

    The z-up camera frame is x=right, y=front, z=up, so the physical image axes in
    the z-up basis are right = +x, down = -z, forward = +y. Since the reorientation
    is `reorient = reorient_zup @ R_conv`, the image axes in the TARGET frame are
    the rows of R_conv (R_conv maps z-up coords -> target coords):
      right   = R_conv[0, :]
      down    = -R_conv[2, :]
      forward = R_conv[1, :]
    This works for any convention rotation, including the z-up convention itself
    (R_conv = identity) and conventions whose +Z points backward (e.g. OpenGL).
    """
    rot = _resolve_convention_rotation(camera_convention)[:3, :3]
    return rot[0, :], -rot[2, :], rot[1, :]


def _zup_reorientation(orientation_type: str) -> np.ndarray:
    """Reorientation to the canonical z-up camera frame (x=right, y=front, z=up).

    Rows are the z-up camera X/Y/Z axes expressed in world (RAS) coordinates.
    """
    pose = _ORIENTATION_POSE[orientation_type]
    reorient = np.eye(4)
    reorient[:3, :3] = np.stack([
        _axis_vec(pose["right"]),
        _axis_vec(pose["front"]),
        _axis_vec(pose["up"]),
    ])
    return reorient


def get_reorientation(
    orientation_type: str = "AP",
    camera_convention: "str | dict[str, str]" = "pytorch3d",
) -> np.ndarray:
    """
    Get the 4x4 reorientation matrix (world -> camera axes) for a C-arm orientation
    (AP/PA) and a camera coordinate convention.

    The derivation is:
      1. Build the reorientation to the canonical z-up camera frame
         (x=right, y=front, z=up), where "front" is the X-ray beam direction
         (AP -> -Y, PA -> +Y).
      2. Right-multiply the convention rotation (align to the z-up camera first,
         then relabel its axes into the target convention):
             reorient = reorient_zup @ R_conv

    camera_convention may be:
      - a name: "pytorch3d" | "diffdrr" | "opengl" | "nerf" | "colmap" | "opencv" | "gs"
      - an explicit dict {"x": ..., "y": ..., "z": ...} mapping each camera axis to a
        signed direction among "right", "front", "up" (z-up basis),
        e.g. {"x": "-right", "y": "up", "z": "front"}.

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
    reorient_zup = _zup_reorientation(orientation_type)
    rot_conv = _resolve_convention_rotation(camera_convention)
    return reorient_zup @ rot_conv  # right-multiply: z-up alignment, then relabel axes


def get_view_matrix(
    alpha: float,
    beta: float,
    sod: float,
    orientation_type: str = "AP",
    camera_convention: "str | dict[str, str]" = "pytorch3d",
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
        View Matrix: (N, 4, 4) array of view matrices [R, T; 0, 1] in world coordinate system
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
    camera_convention: "str | dict[str, str]" = "pytorch3d",
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
    camera_convention: "str | dict[str, str]" = "pytorch3d",
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
        intrinsic_matrix: (4, 4) array of intrinsic matrix K in world coordinate system
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
    camera_convention: "str | dict[str, str]" = "pytorch3d"

    def __post_init__(self):
        self.intrinsic_matrix = get_intrinsic_matrix(
            self.sdd, self.width, self.height, self.delx, self.dely,
            self.x0, self.y0, self.camera_convention
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
    def from_json(json_path: Path) -> "Projector":
        print(f"Loading projector from JSON: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)

        c_arm_geom = data.get("c_arm_geometry", {})

        return Projector(
            sdd=c_arm_geom["sdd"],
            sod=c_arm_geom["sod"],
            width=c_arm_geom["width"],
            height=c_arm_geom["height"],
            delx=c_arm_geom["delx"],
            dely=c_arm_geom["dely"],
            x0=c_arm_geom.get("x0", 0.0),
            y0=c_arm_geom.get("y0", 0.0),
            orientation_type=c_arm_geom.get("orientation_type", "AP"),
            camera_convention=data.get("camera_convention", "pytorch3d")
        )


    def get_projection_matrix(
        self,
        alpha: float,
        beta: float
    ) -> np.ndarray:
        """
        Get the projection matrix K * View in world coordinate system.

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
        Get the projection matrices K * View in world coordinate system for a batch of camera poses.

        Args:
            alphas: (N,) array of rotation angles around Z-axis in radians
            betas: (N,) array of rotation angles around X-axis in radians
        """
        view_matrices = get_view_matrixs(alphas, betas, self.sod, self.orientation_type, self.camera_convention)  # (N, 4, 4)
        M = self.intrinsic_matrix @ view_matrices  # (N, 4, 4)
        return M[:, :3, :]  # (N, 3, 4)


def data_json_to_projection_metrics(
    json_path: Path,
    orientation_type: "str | None" = None,
    camera_convention: "str | dict[str, str] | None" = None,
) -> np.ndarray:
    """
    Get the projection matrices K * View in world coordinate system from a JSON file.

    Args:
        json_path: Path to the JSON file containing the camera parameters and poses.
        orientation_type: optional override for the C-arm orientation (AP/PA); if
            None, read from the JSON (c_arm_geometry.orientation_type, default "AP").
        camera_convention: optional override for the camera convention; if None, read
            from the JSON ("camera_convention", default "pytorch3d").
    Returns:
        projection_matrices: (N, 3, 4) array of projection matrices K * View in world coordinate system
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    c_arm_geom = data.get("c_arm_geometry", {})

    projector = Projector(
        sdd=c_arm_geom["sdd"],
        sod=c_arm_geom["sod"],
        width=c_arm_geom["width"],
        height=c_arm_geom["height"],
        delx=c_arm_geom["delx"],
        dely=c_arm_geom["dely"],
        x0=c_arm_geom.get("x0", 0.0),
        y0=c_arm_geom.get("y0", 0.0),
        orientation_type=orientation_type or c_arm_geom.get("orientation_type", "AP"),
        camera_convention=camera_convention or data.get("camera_convention", "pytorch3d")
    )

    frames = data.get("frames", [])
    alphas_degree = np.array([frame["alpha_degree"] for frame in frames])
    beta_degree = np.array([frame["beta_degree"] for frame in frames])
    alphas = np.deg2rad(alphas_degree)
    betas = np.deg2rad(beta_degree)

    return projector.get_projection_matrices(alphas, betas)  # (N, 3, 4)


def test_projector(
    labe_nii_path: Path = Path(r"test_label.nii.gz"),
    json_path: Path = Path(r"rotate_dsa.json"),
    out_dir: Path = Path(r"temp"),
    n_max: int = 30,
    down_sample_stride: int = 50,
    frame_stride: int = 10
) -> None:
    """
    Test the Projector class by projecting a 3D mesh to 2D images and saving them as PNG files.

    Args:
        mesh_path: Path to the 3D mesh file in VTP format.
        json_path: Path to the JSON file containing the camera parameters and poses.
        out_dir: Directory to save the projected images as PNG files.
    """
    import nibabel as nib

    # Load 3D label from NIfTI file
    label_nii = nib.load(labe_nii_path)
    assert isinstance(label_nii, nib.Nifti1Image), "label_nii must be a Nifti1Image"
    data = label_nii.get_fdata()
    affine = label_nii.affine
    assert affine is not None

    def apply_affine(aff: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to a set of points.

        Args:
            aff: (4, 4) affine transformation matrix
            points: (N, 3) array of points in voxel coordinates or (3,)

        Returns:
            (N, 3) array of points in world coordinates
        """
        if points.ndim == 1:
            points = points.reshape(1, 3)
        A = aff[:3, :3]
        T = aff[:3, 3]
        res = points @ A.T + T

        if res.shape[0] == 1:
            return res.flatten()
        return res

    center_voxel = np.array(data.shape) / 2.0
    center_world = apply_affine(affine, center_voxel)  # (3,)
    affine_centered = affine.copy()
    affine_centered[:3, 3] -= center_world

    label_points = np.argwhere(data > 0)[::down_sample_stride]  # (N, 3)
    label_points_world = apply_affine(affine_centered, label_points)  # (N, 3)
    print(f"Loaded {label_points_world.shape[0]} label points in world coordinates.")

    # Load projector from JSON
    projector = Projector.from_json(json_path)

    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get frames from JSON
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])[::frame_stride]

    # Project and save images for each frame
    for i, frame in enumerate(frames):
        if i >= n_max:
            break
        print(f"Projecting frame {i} with alpha={frame['alpha_degree']}°, beta={frame['beta_degree']}°")
        alpha_degree = frame["alpha_degree"]
        beta_degree = frame["beta_degree"]
        alpha = np.deg2rad(alpha_degree)
        beta = np.deg2rad(beta_degree)

        png_path = out_dir / f"projection_{i:03d}.png"
        projector.project_to_png(alpha, beta, label_points_world, png_path)


if __name__ == "__main__":
    # Example usage
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Project 3D mesh to 2D images using camera parameters from JSON.")
    parser.add_argument("--json_path", type=Path, required=True, help="Path to the JSON file containing camera parameters and poses.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Path to save matric as npy or json.")
    parser.add_argument("--orientation", type=str, default="AP", choices=["AP", "PA"], help="C-arm orientation type (AP or PA).")
    parser.add_argument("--camera_convention", type=str, default="pytorch3d", help="Camera coordinate convention, including pytorch3d, opencv, diffdrr, nerfstudio, gs (default: pytorch3d).")

    args = parser.parse_args()
    json_path = args.json_path
    output_path = args.output

    # Load projector from JSON
    projector = Projector.from_json(json_path)
    projector.orientation_type = args.orientation
    projector.camera_convention = args.camera_convention

    output_path.parent.mkdir(parents=True, exist_ok=True)

    projection_matrices = data_json_to_projection_metrics(
        json_path,
        orientation_type=args.orientation,
        camera_convention=args.camera_convention,
    )
    if output_path.suffix == ".npy":
        # Save projection matrices to .npy file
        np.save(output_path, projection_matrices)
    elif output_path.suffix == ".json":
        # Save projection matrices to .json file
        with open(output_path, "w") as f:
            json.dump(projection_matrices.tolist(), f)
    else:
        raise ValueError("Output path must have .npy or .json extension.")
