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
#   - 相机坐标系: OpenGL 约定, 相机看向 -Z 方向
#   - 图像坐标系: DICOM/OpenCV 约定, 原点在左上角, u 向右, v 向下
#
# 投影公式：
#   p = K @ View @ P_w                 (齐次坐标, 4×4)
#   u = p[0] / p[2],  v = p[1] / p[2]   (透视除法)
#   v = height - v                       (Y 轴翻转 → DICOM)
#
# 内参矩阵 K (4×4)：
#   K = [[fx,  0, -cx,  0],
#        [ 0, fy, -cy,  0],
#        [ 0,  0,  -1,  0],
#        [ 0,  0,   0,  1]]
#   其中 fx = sdd/delx, fy = sdd/dely, cx = width/2, cy = height/2
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

def get_view_matrix(
    alpha: float,
    beta: float,
    sod: float,
) -> np.ndarray:
    """
    Camera follows OpenGL convention, where the camera looks down the negative Z-axis,
    with the Y-axis pointing up and the X-axis pointing to the right.
    Args:
        alpha: float: rotation angle around Z-axis in radians
        beta: float: rotation angle around X-axis in radians
        sod: float: source to object distance in world coordinate system
    Returns:
        View Matrix: (N, 4, 4) array of view matrices [R, T; 0, 1] in world coordinate system
    """
    reorient = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    translate = np.eye(4)
    translate[:3, 3] = np.array([0.0, sod, 0.0])
    rot = euler_angles_to_matrix("ZXY", np.array([alpha, beta, 0.0]))
    m_c2w = rot @ translate @ reorient  # [R_c2w, T_c2w; 0, 1]
    m_w2c = np.linalg.inv(m_c2w)        # [R_c2w^T, -R_c2w^T*T_c2w; 0, 1]
    return m_w2c


def get_view_matrixs(
    alphas: np.ndarray,
    betas: np.ndarray,
    sod: float,
) -> np.ndarray:
    """
    Get the view matrices for a batch of camera poses.

    Args:
        alphas: (N,) array of rotation angles around Z-axis in radians
        betas: (N,) array of rotation angles around X-axis in radians
        sod: float: source to object distance in world coordinate system

    Returns:
        view_matrices: (N, 4, 4) array of view matrices [R, T; 0, 1] in world coordinate system
    """
    reorient = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    translate = np.eye(4)
    translate[:3, 3] = np.array([0.0, sod, 0.0])
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
    y0: float = 0.0
) -> np.ndarray:
    """
    Camera follows OpenGL convention, where the camera looks down the negative Z-axis,
    with the Y-axis pointing up and the X-axis pointing to the right.

    Args:
        sdd: float: source to detector distance
        width: int: detector width in pixels
        height: int: detector height in pixels
        delx: float: pixel size in x direction
        dely: float: pixel size in y direction
        x0: float: x coordinate of the principal point in pixels
        y0: float: y coordinate of the principal point in pixels
    Returns:
        intrinsic_matrix: (4, 4) array of intrinsic matrix K in world coordinate system
    """
    fx = sdd / delx
    fy = sdd / dely
    cx = width / 2.0 + x0
    cy = height / 2.0 + y0
    return np.array([
        [fx, 0, -cx, 0],
        [0, fy, -cy, 0],
        [0, 0,  -1,  0],     # object is in the negative Z direction in world coordinate system, so we need to flip the sign of z
        [0, 0,   0,  1]
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

    def __post_init__(self):
        self.intrinsic_matrix = get_intrinsic_matrix(
            self.sdd, self.width, self.height, self.delx, self.dely,
            self.x0, self.y0
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
        view_matrix = get_view_matrix(alpha, beta, self.sod)    # (4, 4)
        # Project 3D points to 2D points in homogeneous coordinates
        point_3d_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)  # (N, 4)
        point_2d_h = (self.intrinsic_matrix @ view_matrix @ point_3d_h.T).T  # (N, 4)  (x, y, z, w=1)
        # depth = point_2d_h[:, 2] = -Z_camera (positive for points in front of camera)
        depth = point_2d_h[:, 2]
        # Convert to pixel coordinates, u = x / z, v = y / z
        u = point_2d_h[:, 0] / depth
        v = point_2d_h[:, 1] / depth

        v = self.height - v  # Flip v to match DICOM/OpenCV convention
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
            y0=c_arm_geom.get("y0", 0.0)
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
        M = self.intrinsic_matrix @ get_view_matrix(alpha, beta, self.sod)  # (4, 4)
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
        view_matrices = get_view_matrixs(alphas, betas, self.sod)  # (N, 4, 4)
        M = self.intrinsic_matrix @ view_matrices  # (N, 4, 4)
        return M[:, :3, :]  # (N, 3, 4)


def data_json_to_projection_metrics(json_path: Path) -> np.ndarray:
    """
    Get the projection matrices K * View in world coordinate system from a JSON file.

    Args:
        json_path: Path to the JSON file containing the camera parameters and poses.
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
        y0=c_arm_geom.get("y0", 0.0)
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
    out_dir: Path = Path(r"temp_opengl"),
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

    args = parser.parse_args()
    json_path = args.json_path
    output_path = args.output

    # Load projector from JSON
    projector = Projector.from_json(json_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    projection_matrices = data_json_to_projection_metrics(json_path)
    if output_path.suffix == ".npy":
        # Save projection matrices to .npy file
        np.save(output_path, projection_matrices)
    elif output_path.suffix == ".json":
        # Save projection matrices to .json file
        with open(output_path, "w") as f:
            json.dump(projection_matrices.tolist(), f)
    else:
        raise ValueError("Output path must have .npy or .json extension.")
