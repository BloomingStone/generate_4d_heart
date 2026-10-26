import math

import pyvista as pv
import numpy as np
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)
from pytorch3d.structures import Meshes

from .artifacts import Artifact
from .rotate_drr import CArmGeometry



class Torch3DLabelRenderer:
    def __init__(self, c_arm: CArmGeometry):
        self.c_arm_geometry = c_arm
        self.width = int(c_arm.width)
        self.height = int(c_arm.height)
        # PyTorch3D FoVPerspectiveCameras treats scalar fov as VERTICAL fov,
        # and computes horizontal extent via max_x = max_y * aspect_ratio.
        # Compute vertical FOV from detector height to match DRR projection.
        self.fov = 2 * math.atan(c_arm.height * c_arm.dely / 2 / c_arm.sdd) / math.pi * 180
        self.aspect_ratio = c_arm.delx / c_arm.dely
        self.zfar = c_arm.sdd*1.2
        self.raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,          # naive rasterization for fine geometry (stent struts)
        )

        self.artifacts_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.0,
            faces_per_pixel=2,  # front + back faces
            bin_size=0,          # naive rasterization for fine geometry (stent struts)
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            raise RuntimeError("No CUDA device available for Torch3DLabelRenderer")

        # PyTorch3D 默认相机坐标系：
        #   +Z 为前方，+Y 为上方
        # 而这里的工程约定希望：
        #   -Y 为前方，+Z 为上方 (射线源在患者前方，朝向患者后方)
        #
        # 因此需要一个固定的重定向旋转，把外部几何坐标系映射到 PyTorch3D 相机约定。
        if c_arm.orientation_type == "AP":
            reorient_rot = [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]
        elif c_arm.orientation_type == "PA":
            reorient_rot = [
                [-1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]
        else:
            raise ValueError(f"Unsupported orientation_type: {c_arm.orientation_type}")
        self.reorient_rot = torch.tensor(
            reorient_rot,
            dtype=torch.float32,
            device=self.device
        )

    def _get_R_T_by_angles(self, alpha_rad: torch.Tensor, beta_rad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute camera-to-world rotation matrix and translation vector from given angles.
        alpha: rotation around Z-axis (in radians)
        beta: rotation around X-axis (in radians)
        Returns:
            R_c2w: (..., 3, 3) rotation matrix
            T_c2w: (..., 3,) translation vector (Camera position in world coordinates)
        """
        alpha_rad = alpha_rad.to(device=self.device, dtype=self.reorient_rot.dtype)
        beta_rad = beta_rad.to(device=self.device, dtype=self.reorient_rot.dtype)
        cos_a = torch.cos(alpha_rad)
        sin_a = torch.sin(alpha_rad)
        cos_b = torch.cos(beta_rad)
        sin_b = torch.sin(beta_rad)

        R_world = torch.stack([
            torch.stack([cos_a, -cos_b * sin_a,  sin_b * sin_a], dim=-1),
            torch.stack([sin_a,  cos_a * cos_b, -cos_a * sin_b], dim=-1),
            torch.stack([torch.zeros_like(cos_a),   sin_b,           cos_b],        dim=-1),
        ], dim=-2)

        R_c2w = R_world @ self.reorient_rot
        sod = self.c_arm_geometry.sod
        source_y = sod if self.c_arm_geometry.orientation_type == "AP" else -sod
        source = torch.tensor([0.0, source_y, 0.0], device=self.device, dtype=R_world.dtype)
        T_c2w = torch.einsum("...mn,n->...m", R_world, source)
        return R_c2w, T_c2w


    @torch.no_grad()
    def render_silhouette_depth(
        self,
        mesh_pv: pv.PolyData,
        R: torch.Tensor,
        T: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project a mesh to 2D image plane and return silhouette and depth map.
        R: (N, 3, 3) or (3, 3) camera-to-world rotation matrix
        T: (N, 3) or (3,) camera position in world coordinates (T_c2w)
        Returns:
            silhouette: (H, W) binary mask of the projected mesh
            depth: (H, W) depth map of the projected mesh (in world coordinates)
        """
        mesh_pv_tri = mesh_pv.triangulate()
        verts = torch.from_numpy(np.array(mesh_pv_tri.points)).float()
        faces_np = np.array(mesh_pv_tri.faces.reshape(-1, 4)[:, 1:])  # skip the first number which is the number of points per face
        faces = torch.from_numpy(faces_np).long()
        mesh = Meshes([verts.to(self.device)], [faces.to(self.device)])
        if R.dim() == 2:
            R = R[None]
        if T.dim() == 1:
            T = T[None]

        # pytorch 3D has uses the same camera convention as DiffDRR, so we can directly use R_c2w and T_c2w without any conversion
        R_w2c = R  #  pytorch3D 默认使用行向量，因此这里输入的 作用于列向量的 R_c2w 可以不用转置，直接视为R_w2c 矩阵
        T_w2c = -torch.einsum("bmn,bn->bm", (R.transpose(-2, -1), T))  # T_w2c = -R^T @ T_c2w, 对单独的T, 行向量/列向量没有区别

        cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=self.fov,
            aspect_ratio=self.aspect_ratio,
            R=R_w2c.to(self.device),      # pytorch3D 默认使用行向量，因此这里输入的 作用于列向量的 R_c2w 可以不用转置，直接视为R_w2c 矩阵
            T=T_w2c.to(self.device),
            zfar=self.zfar
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )

        # rasterize
        fragments = rasterizer(mesh)
        depth = fragments.zbuf[0, ..., 0]  # depth buffer
        # silhouette
        silhouette = (fragments.pix_to_face[..., 0] >= 0).float()
        return silhouette.cpu().numpy(), depth.cpu().numpy()


    @torch.no_grad()
    def render_silhouette_depth_by_angles(
        self,
        mesh_pv: pv.PolyData,
        alpha: torch.Tensor,
        beta: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        R_c2w, T_c2w = self._get_R_T_by_angles(alpha, beta)
        return self.render_silhouette_depth(mesh_pv, R_c2w, T_c2w)


    @torch.no_grad()
    def render_line_integral(
        self,
        artifacts: list[Artifact],
        R: torch.Tensor,
        T: torch.Tensor,
    ) -> np.ndarray:
        """Compute the total line integral contribution from all artifacts.

        Parameters
        ----------
        artifacts : list[Artifact]
            Artifacts to render.
        R : torch.Tensor
            Rotation matrix (N, 3, 3) or (3, 3) -- camera-to-world rotation.
        T : torch.Tensor
            Translation vector (N, 3) or (3,) -- camera position in world coordinates.

        Returns
        -------
        np.ndarray
            Shape (H, W), accumulated ``sum_i mu_i * path_length_i`` per pixel.
            Zero where no artifact projects.
        """
        total = np.zeros((self.height, self.width), dtype=np.float32)
        if R.dim() == 2:
            R = R[None]
        if T.dim() == 1:
            T = T[None]

        R_w2c = R  # pytorch3D 默认使用行向量，因此这里输入的 作用于列向量的 R_c2w 可以不用转置，直接视为R_w2c 矩阵
        T_w2c = -torch.einsum("bmn,bn->bm", (R.transpose(-2, -1), T))  # T_w2c = -R^T @ T_c2w, 对单独的T, 行向量/列向

        cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=self.fov,
            aspect_ratio=self.aspect_ratio,
            R=R_w2c.to(self.device),
            T=T_w2c.to(self.device),
            zfar=self.zfar,
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.artifacts_settings,
        )

        for art in artifacts:
            path_length = self._render_single(art.mesh, rasterizer)
            total += path_length * art.mu

        return total

    def render_line_integral_by_angles(
        self,
        artifacts: list[Artifact],
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> np.ndarray:
        R_c2w, T_c2w = self._get_R_T_by_angles(alpha, beta)
        return self.render_line_integral(artifacts, R_c2w, T_c2w)

    def _render_single(
        self,
        mesh_pv: pv.PolyData,
        rasterizer: MeshRasterizer,
    ) -> np.ndarray:
        """Render a single mesh and return per-pixel path length.

        For a solid closed mesh with ``faces_per_pixel=2``,
        ``zbuf[..., 0]`` is the front-face depth and ``zbuf[..., 1]``
        is the back-face depth. The path length through the mesh at
        each pixel is ``max(back - front, 0)``.
        """
        mesh_pv_tri = mesh_pv.triangulate()
        verts = torch.from_numpy(np.array(mesh_pv_tri.points)).float()
        faces_np = np.array(mesh_pv_tri.faces.reshape(-1, 4)[:, 1:])
        faces = torch.from_numpy(faces_np).long()
        mesh = Meshes([verts.to(self.device)], [faces.to(self.device)])

        fragments = rasterizer(mesh)
        front_depth = fragments.zbuf[0, ..., 0]  # nearest face  (H, W)
        back_depth = fragments.zbuf[0, ..., 1]  # second-nearest face (back)  (H, W)
        mask = (fragments.pix_to_face[0, ..., 0] >= 0)  # pixels where mesh projects  (H, W)

        path_length = (back_depth - front_depth).clamp(min=0.0)
        path_length = path_length.cpu().numpy()
        path_length[~mask.cpu().numpy()] = 0.0

        return path_length
