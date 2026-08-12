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

    @torch.no_grad()
    def render_silhouette_depth(
        self, 
        mesh_pv: pv.PolyData, 
        R: torch.Tensor, 
        T: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        mesh_pv_tri = mesh_pv.triangulate()
        verts = torch.from_numpy(np.array(mesh_pv_tri.points)).float()
        faces_np = np.array(mesh_pv_tri.faces.reshape(-1, 4)[:, 1:])  # skip the first number which is the number of points per face
        faces = torch.from_numpy(faces_np).long()
        mesh = Meshes([verts.to(self.device)], [faces.to(self.device)])

        # pytorch 3D has uses the same camera convention as DiffDRR, so we can directly use R_c2w and T_c2w without any conversion
        R = R.squeeze()
        T = T.squeeze()
        cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=self.fov,
            aspect_ratio=self.aspect_ratio,
            R=R[None].to(self.device),      # pytorch3D 默认使用行向量，因此这里输入的 作用于列向量的 R_c2w 可以不用转置，直接视为R_w2c 矩阵
            T=(-R.T @ T)[None].to(self.device), 
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
            Rotation matrix (3, 3) -- camera-to-world rotation.
        T : torch.Tensor
            Translation vector (3,) -- camera position in world coordinates.

        Returns
        -------
        np.ndarray
            Shape (H, W), accumulated ``sum_i mu_i * path_length_i`` per pixel.
            Zero where no artifact projects.
        """
        total = np.zeros((self.height, self.width), dtype=np.float32)

        R = R.squeeze()
        T = T.squeeze()

        cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=self.fov,
            aspect_ratio=self.aspect_ratio,
            R=R[None].to(self.device),
            T=(-R.T @ T)[None].to(self.device),
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

