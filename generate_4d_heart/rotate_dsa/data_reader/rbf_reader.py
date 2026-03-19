from typing import Literal
from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from monai.networks.blocks.warp import Warp
from einops import rearrange, einsum
from cupyx.scipy import ndimage
import cupy as cp

from generate_4d_heart.roi import ROI
from generate_4d_heart.rotate_dsa.data_reader.data_reader import (
    DataReader, DataReaderResult, Coronary, separate_coronary, get_coronary_centering_affine,
    load_nifti, get_mesh_in_voxel, apply_affine, get_mesh_in_world
)
from generate_4d_heart.ssm import SSMReader, polydata_io, Landmark
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.rotate_dsa.types import CoronaryType
from generate_4d_heart import CavityLabel


logger = logging.getLogger(__name__)

@dataclass
class _Data:
    device: torch.device
    image: Tensor
    cavity: Tensor
    affine: np.ndarray
    lca: Tensor
    rca: Tensor
    
    lca_centering_affine: np.ndarray = field(init=False)
    rca_centering_affine: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.lca_centering_affine = get_coronary_centering_affine(self.lca, self.affine, self.device)
        self.rca_centering_affine = get_coronary_centering_affine(self.rca, self.affine, self.device)
    
    @property
    def all_coronary_label(self) -> Tensor:
        res = self.lca.clone().to(torch.uint8)
        res += self.rca
        return res


@dataclass
class RBFReader(DataReader):
    image_nii: Path
    cavity_nii: Path
    coronary_nii: Path
    ssm_reader: SSMReader= field(default_factory=SSMReader)
    device: torch.device = field(default_factory= lambda: torch.device("cuda:0"))
    num_components_used: int = 1
    motion_multiplier: float = 1.3
    attenuation_transition: float = 5.0

    def __post_init__(self):
        print(f"\nInitializing RBFReader with \n\t- image_nii={self.image_nii} \n\t- cavity_nii={self.cavity_nii} \n\t- coronary_nii={self.coronary_nii}")
        self.n_phases = self.ssm_reader.n_phases
        
        logger.info(f"Loading data from {self.image_nii}, {self.cavity_nii}, {self.coronary_nii}")
        image, affine = load_nifti(self.image_nii)
        assert affine is not None
        cavity, cavity_affine = load_nifti(self.cavity_nii, is_label=True)
        cavity_np = cavity.squeeze().cpu().numpy()
        coronary, coronary_affine = load_nifti(self.coronary_nii, is_label=True)
        assert np.allclose(affine, cavity_affine)
        assert np.allclose(affine, coronary_affine)
        
        lca, rca = separate_coronary(coronary, self.device)        
        self.origin_data = _Data(
            self.device, image, cavity, affine, 
            lca, rca
        )
        _, pericardium_mask = generate_pericardium(cavity_np, coronary.squeeze().cpu().numpy())

        self._origin_volume_size = image.shape[2:]   #type: ignore
        self._origin_volume_affine = affine
        
        logger.info("ROI initialized from cavity")
        self.roi = ROI.get_from_cavity_np(cavity_np, affine, padding=30)
        def crop(x: Tensor) -> Tensor:
            return self.roi.crop_on_data(x.clone())
        
        self.cropped_data = _Data(
            self.device, crop(image), crop(cavity), self.roi.affine_after_crop,
            crop(lca), crop(rca)
        )
        
        logger.info("Extracting coronary meshes in world coordinate")
        self.coronary_mesh_world = {
            CoronaryType.LCA: get_mesh_in_world(lca, affine, self.device),
            CoronaryType.RCA: get_mesh_in_world(rca, affine, self.device)
        }
        
        pericardium_mask = self.roi.crop_on_data(pericardium_mask)
        with cp.cuda.Device(self.device.index):
            external_dist = ndimage.distance_transform_edt(1 - cp.asarray(pericardium_mask))
            attenuation_mask = cp.asnumpy(cp.exp(-external_dist / self.attenuation_transition))  # 从心包外部开始，距离越远运动衰减越强，距离为transition时衰减为1/e #type: ignore
        self.attenuation_mask = torch.from_numpy(attenuation_mask).to(self.device, torch.float32)[None, None]  # (1, 1, *shape_after_crop)
        
        logger.info("Loading SSM result and initializing RBF")
        ssm_result = self.ssm_reader.load(
            cavity=cavity_np,
            affine=cavity_affine,
            num_components_used=self.num_components_used,
            motion_multiplier=self.motion_multiplier
        )
        self.rbf = KDTreeRBF(ssm_result.landmark).to(self.device).eval()
        
        logger.info("Caching grid points for RBF-based warping")
        shape_zoomed = self.roi.shape_after_crop_and_zoom
        affine_zoomed = self.roi.affine_after_crop_and_zoom
        grid_zoomed = np.meshgrid(
            np.arange(shape_zoomed[0]), 
            np.arange(shape_zoomed[1]), 
            np.arange(shape_zoomed[2]), 
            indexing="ij"
        )
        coords_zoomed = np.stack(grid_zoomed, axis=-1)  # (N, 3)
        coords_zoomed = apply_affine(coords_zoomed, affine_zoomed)
        self.coords_zoomed = torch.from_numpy(coords_zoomed).to(self.device, torch.float32)
        
        self.affine_crop = torch.from_numpy(self.roi.affine_after_crop).to(self.device, torch.float32)
        self.affine_crop_inv: torch.Tensor = torch.linalg.inv(self.affine_crop)
        
        self.warp_label = Warp(mode="nearest", padding_mode="border").to(self.device).eval()
        self.warp_image = Warp(mode="bilinear", padding_mode="border").to(self.device).eval()
        
        print("RBFReader initialized successfully.")

    @torch.no_grad()
    def get_data(self, phase: CardiacPhase, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        cor_type = CoronaryType(coronary_type)
        cor_volume = self.cropped_data.lca if cor_type in ("LCA", CoronaryType.LCA) else self.cropped_data.rca
        
        # here use the affine from origin_data as we will recover the cropped and zoomed data back to original space, 
        # so the coronary centering affine should also be match with the original shape
        cor_centering_affine = self.origin_data.lca_centering_affine if cor_type in ("LCA", CoronaryType.LCA) else self.origin_data.rca_centering_affine
        cor_mesh = self.coronary_mesh_world[cor_type].copy()
        mesh_points = torch.from_numpy(cor_mesh.points).to(self.device, torch.float32)
        mesh_points += self.rbf(phase, mesh_points)     # moving in world coordinate
        cor_mesh.points = mesh_points.cpu().numpy()
        ori_T = self._origin_volume_affine[:3, 3]       # ori_T and new_T are also in world coordinate, we use them to adjust the coronary mesh position
        new_T = cor_centering_affine[:3, 3]
        cor_mesh.points += (new_T - ori_T)
        
        dx: torch.Tensor = self.rbf(phase, self.coords_zoomed)  # (*shape_zoomed, 3) in world coordinate
        dx = rearrange(dx, "x y z c -> 1 c x y z")
        dx = F.interpolate(
            dx, 
            size=self.roi.shape_after_crop, 
            mode="trilinear", 
            align_corners=False
        )  # (1, 3, *shape_after_crop)  in world coordinate
        
        R_world_to_voxel = self.affine_crop_inv[:3, :3]        
        dx = einsum(dx, R_world_to_voxel, "b c x y z, c i -> b i x y z")  # in voxel coordinate
        
        dx = dx * self.attenuation_mask  # Attenuate the movement away from the pericardium
        
        dx = invert_displacement_field(dx)  # forward deformation field -> inverse deformation field, still in voxel coordinate
        
        def warp_and_recover(tensor: Tensor, mode: Literal["image", "cavity", "coronary"]) -> Tensor:
            if mode == "image":
                warp = self.warp_image
                recover = lambda x: self.roi.recover_cropped_tensor(x, background=self.origin_data.image)
                final_dtype = torch.float32
            else:
                warp = self.warp_label
                recover = self.roi.recover_cropped_tensor
                if mode == "cavity":
                    final_dtype = torch.uint8
                elif mode == "coronary":
                    final_dtype = torch.bool
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                
            return recover(warp(tensor.to(self.device, torch.float32), dx)).cpu().to(final_dtype)
        
        return DataReaderResult(
            phase=phase,
            volume=warp_and_recover(self.cropped_data.image, mode="image"),     # need shape as (B, C, D, H, W)
            cavity_label=warp_and_recover(self.cropped_data.cavity, mode="cavity"),
            affine=self.cropped_data.affine,
            coronary=Coronary(
                type=cor_type,
                label=warp_and_recover(cor_volume, mode="coronary"),
                centering_affine=cor_centering_affine,
                mesh_in_world=cor_mesh
            )
        )
    
    @torch.no_grad()
    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        return self.get_data(CardiacPhase(0.0), coronary_type)
    
    @torch.no_grad()
    def get_original_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        coronary_type = CoronaryType(coronary_type)
        coronary = self.origin_data.lca if coronary_type in ("LCA", CoronaryType.LCA) else self.origin_data.rca
        coronary_centering_affine = self.origin_data.lca_centering_affine if coronary_type in ("LCA", CoronaryType.LCA) else self.origin_data.rca_centering_affine
        
        return DataReaderResult(
            phase=CardiacPhase(0.0),
            volume=self.origin_data.image[None, None],     # need shape as (B, C, D, H, W)
            cavity_label=self.origin_data.cavity[None, None],
            affine=self.origin_data.affine,
            coronary=Coronary(
                type=coronary_type,
                label=coronary[None, None],
                centering_affine=coronary_centering_affine,
                mesh_in_world=self.coronary_mesh_world[coronary_type].copy()
            )
        )
    
    @property
    def lca_centering_affine(self) -> np.ndarray:
        return self.cropped_data.lca_centering_affine
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        return self.cropped_data.rca_centering_affine
    
    @property
    def volume_size(self) -> tuple[int, int, int]:
        return self.roi.original_shape
    
    @property
    def volume_affine(self) -> np.ndarray:
        return self.roi.original_affine

def generate_pericardium(cavity_data: np.ndarray, coronary_data: np.ndarray, padding_voxel: float=5.0) -> tuple[pv.PolyData, np.ndarray]:
    """
    使用 PyVista 提取并平滑心包包络
    """
    cavity_all_mask = np.isin(cavity_data, list(CavityLabel)).astype(np.uint8)
    cavity_add_coronary_mask = np.logical_or(cavity_all_mask, coronary_data > 0).astype(np.uint8)

    # 2. 提取等值面 (Marching Cubes)
    cavity_all_mesh: pv.PolyData = pv.wrap(cavity_add_coronary_mask)\
        .contour([1], method="flying_edges")\
        .triangulate()\
        .smooth_taubin()\
        .decimate_pro(
            reduction=0.95,          # 减少 三角面片
            preserve_topology=True, # 防止破洞
        )\
        .clean()

    points = cavity_all_mesh.points
    normals = cavity_all_mesh.point_normals
    offset_points = points + normals * padding_voxel

    pericardium_mesh = pv.PolyData(offset_points)\
        .delaunay_3d()\
        .extract_surface(algorithm=None)
    
    ref_volume = pv.ImageData(dimensions=cavity_all_mask.shape)
    pericardium_mask = pericardium_mesh\
        .voxelize_binary_mask(reference_volume = ref_volume)\
        .point_data['mask']\
        .reshape(cavity_all_mask.shape, order='F')
    
    return pericardium_mesh, pericardium_mask.astype(np.uint8)


class KDTreeRBF(nn.Module):
    ctrl_pts: torch.Tensor  # (n_ctrl_pts, 3)
    ctrl_disps: torch.Tensor   # (n_phase, n_ctrl_pts, 3)
    
    def __init__(
        self,
        landmark: Landmark,
        *,
        sigma: float = 10.0,
        chunk_size: int = 1_000_000,
        radius_factor: float = 3.0,
        k: int = 64,
        n_ctrl_pts: int = 500
    ):
        super().__init__()
        interval = max(1, landmark.mesh.n_points // n_ctrl_pts)
        
        ctrl_pts=landmark.mesh.points[::interval]     # 降采样 ctrl point, 加快处理速度
        self.n_ctrl_pts: int = ctrl_pts.shape[0]

        self.n_phases = landmark.deforms.shape[0]
        ctrl_disps = landmark.deforms[:, ::interval]  # (n_phase, n_ctrl_pts, 3)
        
        self.register_buffer("ctrl_pts", torch.from_numpy(ctrl_pts).half())     # (n_ctrl_pts, 3)
        self.register_buffer("ctrl_disps", torch.from_numpy(ctrl_disps).half() )      # (n_phase, n_ctrl_pts, 3)
        
        self.sigma = sigma
        self.chunk_size = chunk_size
        self.radius_factor = radius_factor
        self.k = k
        self.radius = sigma * radius_factor
        
        self.tree = KDTree(ctrl_pts)
        
        self._copy_stream: torch.cuda.Stream|None = None
        self.cached_pts: torch.Tensor|None  = None
    
    def cache_points(self, points: np.ndarray):
        device = self.ctrl_pts.device
        self.cached_pts = torch.from_numpy(points).to(device, torch.half, non_blocking=True)
    
    @property
    def copy_stream(self) -> torch.cuda.Stream|None:
        device = self.ctrl_pts.device
        if device.type == 'cpu':
            return None
        if (self._copy_stream is None or self._copy_stream.device != device) and device.type == 'cuda': 
            self._copy_stream = torch.cuda.Stream(device=device)
        return self._copy_stream
    
    def _update_motion_by_prev(
        self,
        points: torch.Tensor,   # (N, 3)
        disp: torch.Tensor, # (n_ctrl_pts, 3)
        motion: torch.Tensor,   # (N, 3), to be update
        prev: tuple[int, int, torch.Tensor]|None
    ):
        if prev is None:
            return
        
        start, end, idx = prev
        chunk, k = idx.shape

        # waiting for copy
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        
        # for points with neighbors less than k, KDTree will fill its dist by inf 
        # and set idx by tree.n (tree.n == ctrl_pts.shape[0] == n_ctrl_pts)
        mask = (idx != self.n_ctrl_pts)
        idx[~mask] = 0
        
        # fat idx to avoid cache missing， equals to follows
        # ctrl_pts_t = self.ctrl_pts[idx]
        # ctrl_disp_t = disp[idx]
        flat_idx = idx.reshape(-1)
        ctrl_pts_t = self.ctrl_pts.index_select(0, flat_idx)
        ctrl_pts_t = ctrl_pts_t.view(chunk, k, 3)
        ctrl_disp_t = disp.index_select(0, flat_idx)
        ctrl_disp_t = ctrl_disp_t.view(chunk, k, 3)
        points_t = points[start:end]
        
        # gaussian kernel
        diff = points_t.unsqueeze(1) - ctrl_pts_t
        r2 = (diff * diff).sum(-1)
        w = torch.exp(-r2 / (2 * self.sigma**2))
        # we used 0 to fill invalid idx before, i.e. calculate `w` with points_t[0], which need to be removed.
        w = w * mask
        w_sum = w.sum(dim=1, keepdim=True)  # w >= 0, so w_sum >= 0
        w_sum[w_sum < 1e-5] = 1.0  # avoid division by zero. 0 / 1 = 0, so it does not change the result. for value < 1e-5, we consider it as 0
        w = w / w_sum
        
        # w = w / (w.sum(dim=1, keepdim=True) + 1e-5)

        chunk_motion = (w.unsqueeze(-1) * ctrl_disp_t).sum(1)
        motion[start:end] = chunk_motion
    
    def forward(
        self,
        phase: CardiacPhase,
        points: np.ndarray|torch.Tensor|None = None,    # (..., 3)
    ) -> torch.Tensor:
        """Calculate the motion for the given phase and points. Return motion tensor on module's device"""
        idx0 = phase.closest_index_floor(self.n_phases)
        idx1 = phase.closest_index_ceil(self.n_phases)
        w = float(phase) * self.n_phases - idx0
        disp = w * self.ctrl_disps[idx1] + (1 - w) * self.ctrl_disps[idx0]
        
        device = self.ctrl_pts.device
        
        match points:
            case None:
                assert self.cached_pts is not None, "points is None and cached_pts is None, please call cache_points first"
                points_gpu = self.cached_pts
                points_np = points_gpu.cpu().numpy()
            case torch.Tensor():
                points_np = points.cpu().numpy()
                points_gpu = points.to(device, torch.half, non_blocking=True)
            case np.ndarray():
                points_gpu = torch.from_numpy(points).to(device, torch.half, non_blocking=True)
                points_np = points.copy()
            case _:
                raise TypeError(f"Unsupported type for points: {type(points)}")
        
        shape = points_np.shape
        assert shape[-1] == 3, f"Last dimension of points must be 3, but got {shape[-1]}"
        points_np = points_np.reshape(-1, 3)  # (N, 3)
        points_gpu = points_gpu.reshape(-1, 3)  # (N, 3)
        
        N = points_np.shape[0]
        motion = torch.zeros((N, 3), dtype=torch.half, device=device)
        prev :tuple[int, int, torch.Tensor]|None = None
        
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            
            # ---------- CPU KDTree ----------
            # set distance_upper_bound=radius thus points with dist > radius will not return
            # set workers = -1 to use all CPU
            _, idx = self.tree.query(points_np[start:end], k=self.k, distance_upper_bound=self.radius, workers=-1)
            if self.k == 1:
                idx = idx[:, None]  # (chunk, 1)    # type: ignore

            idx_cpu = torch.from_numpy(idx)

            # ---------- async H2D ----------
            with torch.cuda.stream(self.copy_stream):
                idx_gpu = idx_cpu.to(device)

            self._update_motion_by_prev(points_gpu, disp, motion, prev)
            prev = (start, end, idx_gpu)
        
        # process the last `prev` block
        self._update_motion_by_prev(points_gpu, disp, motion, prev)
        
        return motion.float().reshape(shape)

    def inverse_forward(self):
        pass
        

def invert_displacement_field(
    u_forward: torch.Tensor, 
    n_iters: int=10
) -> torch.Tensor:
    """
    The inverse displacement field v is solved using a fixed-point iteration method such that for each voxel coordinate y, we have v(y) = -u(y + v(y)). 
    u_forward: Forward displacement field Tensor of shape (W, H, D, 3) in units of voxels.
    n_iters: number of iterations.
    """
    B, C, D, H, W = u_forward.shape
    assert C == 3, f"Expected the last dimension of u_forward to be 3, but got {C}"
    u = u_forward
    device = u.device
    
    # 2. 生成标准的采样网格 [-1, 1]
    grid_d, grid_h, grid_w = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    identity_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0) # (1, D, H, W, 3)

    # 3. u 是体素单位，需要转成 [-1, 1] 相对单位
    u_norm = u.clone()
    u_norm[:, 0] = u_norm[:, 0] / (W / 2.0)
    u_norm[:, 1] = u_norm[:, 1] / (H / 2.0)
    u_norm[:, 2] = u_norm[:, 2] / (D / 2.0)

    # 4. 固定点迭代
    v = -u_norm # 初始猜测
    for _ in range(n_iters):
        # 计算当前 y + v(y) 应该去哪里采样 u
        sample_coords = identity_grid + v.permute(0, 2, 3, 4, 1)
        # 迭代公式: v = -u(y + v)
        v = -F.grid_sample(u_norm, sample_coords, mode='bilinear', padding_mode='border', align_corners=True)
    
    # 5. 还原 v 到体素单位
    v[:, 0] *= (W / 2.0)
    v[:, 1] *= (H / 2.0)
    v[:, 2] *= (D / 2.0)
    
    return v
