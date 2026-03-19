# %% [markdown]
# # 测试使用径向基函数（RBF）生成冠脉mesh运动与密集形变场

# %% [markdown]
# 
# ## 导入与读取数据

# %%
from pathlib import Path
import numpy as np
import nibabel as nib
import pyvista as pv
from nibabel.nifti1 import Nifti1Image

from generate_4d_heart.moving_dvf.utils import MaybeFlipTransform

# import logging
# 强制重新配置，设置级别为 DEBUG
# logging.basicConfig(level=logging.DEBUG, force=True)


root_dir = Path.cwd().parent

cavity_path = root_dir / "test" / "test_data" / "volume_with_dvf" / "cavity.nii.gz"
coronary_path = root_dir / "test" / "test_data" / "volume_with_dvf" / "coronary.nii.gz"

ssm_dir = root_dir / "generate_4d_heart" / "ssm_world"
ssm_template_path = ssm_dir / "ssm_template.vtk"
b_motion_path = ssm_dir / "b_motion_mean_per_phase.npy"
P_motion_path = ssm_dir / "P_motion.npy"

(temp_output := Path("temp")).mkdir(parents=True, exist_ok=True)


def read_nii(path: Path) -> tuple[Nifti1Image, np.ndarray]:
    img = nib.load(path)
    assert isinstance(img, Nifti1Image)
    return img, img.get_fdata()

cavity_img, cavity_data = read_nii(cavity_path)
coronary_img, coronary_data = read_nii(coronary_path)

ssm_template: pv.PolyData = pv.read(ssm_template_path)  #type: ignore
assert isinstance(ssm_template, pv.PolyData)
b_motion: np.ndarray = np.load(b_motion_path)
P_motion: np.ndarray = np.load(P_motion_path)


# %% [markdown]
# b_motion.shape = (5, 20, 7) ->  (n_labels, n_phases, n_components)
# 
# P_motion.shape = (5, 7, 500, 3) -> (n_labels, n_components, n_points, n_dim)

# %%
print(b_motion.shape, P_motion.shape)

n_labels, n_phases, n_components = b_motion.shape
n_labels_, n_components_, n_points, n_dim = P_motion.shape

assert n_labels == n_labels_ and n_components == n_components_ and n_dim == 3


# %% [markdown]
# ## 提取冠脉中心线和表面mesh

# %%
def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply affine transformation to points with shape (..., 3)."""
    assert points.shape[-1] == 3, "The last dimension of points must be 3."
    assert affine.shape == (4, 4), "affine must be shape (4, 4)."
    return points @ affine[:3, :3].T + affine[:3, 3]


# %%
from skimage.morphology import skeletonize

center_line_points = np.argwhere(skeletonize(coronary_data)).astype(np.float32)
assert coronary_img.affine is not None
center_line_points = apply_affine(center_line_points, coronary_img.affine)
cl_clouds = pv.PolyData(center_line_points)

coronary_mesh: pv.PolyData = pv.wrap(coronary_data).contour([1], method="flying_edges").triangulate().smooth_taubin(n_iter=50).clean()
coronary_mesh.points = apply_affine(coronary_mesh.points, coronary_img.affine)

# %% [markdown]
# ## 获取landmark
# 
# 从 nii 标注获取表面label_surface，并将smm_template变形配准到label_surface, 获取landmarks

# %%
import cupy as cp
import pyacvd
from cupyx.scipy import ndimage
from generate_4d_heart.moving_dvf.ssm import deform_surface
from generate_4d_heart import NUM_TOTAL_POINTS, CavityLabel


def _get_largest_connected_component(data):
    """
    Obtain the largest connected region in the binary image. Copy from Phantom/script/get_surface_cloud.py
    Args:
        data: the binary image
    Returns:
        np.ndarray: the largest connected region in the binary image
    """
    data_cp = cp.asarray(data)
    labeled_data, num_features = ndimage.label(data_cp)  # type: ignore
    if num_features == 1:
        return data_cp
    sizes = ndimage.sum(data_cp, labeled_data)
    largest_component = sizes.argmax()
    return (labeled_data == largest_component).astype(cp.uint8)


def get_surface_from_label(
        label: np.ndarray,
        affine: np.ndarray
) -> pv.PolyData:
    """
    Get the cloud from the label. Copy from Phantom/script/get_surface_cloud.py
    Args:
        nii_path: the path to the nii file
        max_point_num: the maximum number of points in the cloud for each label
    Returns:
        pv.PolyData: the cloud
        np.ndarray: the affine matrix of the nii file
    """
    surface_all = pv.PolyData()
    
    label_cp = cp.asarray(label)
    for label_id in CavityLabel:
        assert label_id > 0
        if label_id == CavityLabel.LV_MYO:
            # The myocardial part of the left ventricle is combined with the left ventricular cavity part
            mask = cp.zeros(label_cp.shape, dtype=cp.uint8)
            mask[label_cp == CavityLabel.LV_MYO] = 1
            mask[label_cp == CavityLabel.LV] = 1
            mask = (mask > 0).astype(cp.uint8)
        else:
            mask = (label_cp == label_id).astype(cp.uint8)
        
        structure = ndimage.generate_binary_structure(3, 3)
        mask = ndimage.binary_closing(mask, iterations=1, structure=structure)
        mask = ndimage.binary_opening(mask, iterations=1, structure=structure)
        mask = _get_largest_connected_component(mask)
        mask: np.ndarray = cp.asnumpy(mask)

        surface = pv.wrap(mask).contour([1], method="flying_edges").triangulate().smooth_taubin().clean()
        cluster = pyacvd.Clustering(surface)
        cluster.cluster(NUM_TOTAL_POINTS)
        surface = cluster.create_mesh().triangulate().clean()
        assert isinstance(surface, pv.PolyData)
        if np.isnan(surface.points).any():
            raise ValueError(f"NaN in points")
        if not surface.is_manifold:
            raise ValueError(f"Mesh is not manifold")
        surface.point_data["label"] = np.ones(surface.n_points).astype(np.uint8) * label_id
        surface.cell_data["label"] = np.ones(surface.n_cells).astype(np.uint8) * label_id
        surface_all = surface_all.merge(surface)
    
    assert isinstance(surface_all, pv.PolyData)
    surface_all.points = apply_affine(surface_all.points, affine)
    return surface_all

assert cavity_img.affine is not None
label_surface = get_surface_from_label(cavity_data, cavity_img.affine)
landmark_surface = deform_surface(ssm_template, label_surface.copy(), 0)
label_surface.save(temp_output / "label_surface.vtk")

# %%
bounds_ssm = np.array(ssm_template.bounds).reshape(3, 2)
size_ssm = np.abs(bounds_ssm[:, 1] - bounds_ssm[:, 0])

bounds_surface = np.array(label_surface.bounds).reshape(3, 2)
size_surface = np.abs(bounds_surface[:, 1] - bounds_surface[:, 0])

size_ratio = size_surface / size_ssm
print("size_ratio:", size_ratio) # 心脏大小因人而异

# %% [markdown]
# ssm_template 和 label_surface 在心腔形状上存在一些区别，但不影响冠脉的运动。
# 虽然可以通过增加迭代次数使得两者尽可能贴合，但可以回导致mesh均匀性变差

# %%

# plotter = pv.Plotter(shape=(1, 3), notebook=False)
# plotter.camera_position = pv.CameraPosition(
#     (-68.16251727594782, 104.066792868929, 344.7257297397248),
#     (0, 0, 0),
#     (0.7523987560658962, -0.5745229979148703, 0.3222102368600382)
# )
# c = np.array(label_surface.center)
# plotter.subplot(0, 0)
# plotter.add_mesh(ssm_template, scalars="label")
# plotter.subplot(0, 1)
# plotter.add_mesh(label_surface.translate(-c), scalars="label")
# plotter.add_mesh(cl_clouds.translate(-c), color="red")
# plotter.subplot(0, 2)
# plotter.add_mesh(landmark_surface.translate(-c), scalars="label")
# plotter.add_mesh(cl_clouds.translate(-c), color="red")
# plotter.link_views()
# plotter.show()

# %% [markdown]
# 将landmark_surface按标签进行拆分，便于后续分别增加 PCA 提取的运动

# %%
template_labels = landmark_surface.cell_data['label']

landmark_submeshes: dict[int, pv.PolyData] = {}
for label_i in range(n_labels):
    landmark_submeshes[label_i] = landmark_surface.extract_cells(
        landmark_surface.cell_data['label'] == label_i+1
    ).extract_surface(algorithm=None)

# %% [markdown]
# ## 应用SSM于landmark，生成运动
# 
# 使用 PCA 中的运动主成分和强度为不同标签的心腔区域施加运动，得到cavity_ij和merge 得到 deformed_cavities。最后存储到list中作为多期相心腔运动结果
# 
# 同时在每个心腔mesh中 用 "deform_00", "deform_01" 等作为key存储不同相位的运动位移。最后将所有landmark_submeshes在一起，存储所有面片几何和位移

# %%
num_components_used = 1     # PCA运动的前几个主成分，增大能还原复杂运动，但也会引入噪声
# 通过尺度比调整 motion zoom rate
motion_zoom_rate = size_ratio.mean() # motion zoomed by size ratio

deformed_cavities: list[pv.PolyData] = []
for phase_j in range(n_phases):
    cavities_list = []
    for label_i in range(n_labels):
        b = b_motion[label_i, phase_j, :num_components_used] * motion_zoom_rate
        P = P_motion[label_i, :num_components_used]
        deformation = np.einsum('K, KMD -> MD', b, P)
        
        cavity_ij = landmark_submeshes[label_i].copy()
        cavity_ij.points += deformation
        cavities_list.append(cavity_ij)

        landmark_submeshes[label_i].point_data[f"deform_{phase_j:02d}"] = deformation
    
    deformed_cavities.append(pv.merge(cavities_list))

landmark_surface: pv.PolyData = pv.merge(list(landmark_submeshes.values()))



# %%
from generate_4d_heart.moving_dvf.ssm import _extract_faces_by_label

def polydata_to_label_nii(
    polydata: pv.PolyData,
    ref_nii: Nifti1Image,
) -> Nifti1Image:
    """
    将 polydata surface（以 label 区分）转换为 label segmentation volumes。
    """
    shape = ref_nii.shape
    affine = ref_nii.affine
    assert affine is not None
    affine_inv = np.linalg.inv(affine)
    polydata = polydata.copy()
    polydata.points = apply_affine(polydata.points, affine_inv)
    
    reference_volume = pv.ImageData(dimensions=shape)
    
    label_volume = np.zeros(shape, dtype=np.uint8)
    labels = np.unique(polydata.point_data["label"]).tolist()
    labels = [int(label) for label in labels if label > 0 and label != 2]
    labels = sorted(labels, reverse=True)
    labels.append(2)   # 左心室部分 label=2， 因为此前和左心肌合并处理，故此处需要最后处理，以覆盖在所有label之上

    for label in labels:
        face_of_label = _extract_faces_by_label(polydata, label)
        mask_image_data = face_of_label.voxelize_binary_mask(reference_volume = reference_volume)
        mask = mask_image_data.point_data['mask'].reshape(shape, order='F')
        label_volume[mask == 1] = label
    
    return Nifti1Image(label_volume, affine)

landmark_surface.save(temp_output / "landmark_surface.vtk")
(surfaces_output_dir := temp_output / "./deformed_surfaces").mkdir(parents=True, exist_ok=True)
(volumes_output_dir := temp_output / "./deformed_volumes").mkdir(parents=True, exist_ok=True)
for index, cavity in enumerate(deformed_cavities):
    cavity.save(surfaces_output_dir / f"{index:02d}.vtk")
    polydata_to_label_nii(cavity, cavity_img).to_filename(volumes_output_dir / f"{index:02d}.nii.gz")
    
    

# %% [markdown]
# ## KDTree 生成形变场
# 
# 用上一部已提取motion的landmark生成径向基函数，已得到各空间点区域的正向形变场
# 
# 使用多种方式加快处理速度：
# - 使用 KDtree，仅提取少量最近点用于计算位移
# - 使用 chunk 小批量统一向量化处理，加快计算速度
# - 使用 copy tream 异步进行 tensor 传输与计算
# - 整合为 nn.Module 一统一移动 device
# - 使用 cache 保存 points, 因为生成运动时原始点保持不变，仅改变 phase。避免重复 copy tensor
# - 使用 `index_select` 替代 advaced indexing (类似 `x[ [[0,1], [1,3]] ]`) 加快处理速度
# 
# 多种方法加速后处理 128^3 的 grid 只需要 0.5 s

# %%
import numpy as np
from typing import TypeVar
import torch
from scipy.spatial import KDTree
from torch import nn
from generate_4d_heart import NUM_TOTAL_PHASE
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase

class KDTreeRBF(nn.Module):
    ctrl_pts: torch.Tensor  # (n_ctrl_pts, 3)
    ctrl_disps: torch.Tensor   # (n_phase, n_ctrl_pts, 3)
    
    def __init__(
        self,
        landmark_surface: pv.PolyData,
        *,
        sigma: float = 30.0,
        chunk_size: int = 5_000_000,
        radius_factor: float = 3.0,
        k: int = 32,
        interval: int = 5
    ):
        super().__init__()
        ctrl_pts=landmark_surface.points[::interval]     # 降采样 ctrl point: 加快处理速度; 2500 -> 500
        self.n_ctrl_pts: int = ctrl_pts.shape[0]
        self.n_phases = NUM_TOTAL_PHASE
        ctrl_disps = np.zeros((self.n_phases, self.n_ctrl_pts, 3))
        for phase_j in range(self.n_phases):
            ctrl_disps[phase_j] = landmark_surface.point_data[f"deform_{phase_j:02d}"][::interval]
        
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
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)

        chunk_motion = (w.unsqueeze(-1) * ctrl_disp_t).sum(1)
        motion[start:end] = chunk_motion
    
    def forward(
        self,
        phase: CardiacPhase,
        points: np.ndarray|torch.Tensor|None = None,
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
        
        N = points_np.shape[0]
        motion = torch.zeros((N, 3), dtype=torch.half, device=device)
        prev :tuple[int, int, torch.Tensor]|None = None
        
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            
            # ---------- CPU KDTree ----------
            # set distance_upper_bound=radius thus points with dist > radius will not return
            # set workers = -1 to use all CPU
            _, idx = self.tree.query(points_np[start:end], k=self.k, distance_upper_bound=self.radius, workers=-1)

            idx_cpu = torch.from_numpy(idx)

            # ---------- async H2D ----------
            with torch.cuda.stream(self.copy_stream):
                idx_gpu = idx_cpu.to(device)

            self._update_motion_by_prev(points_gpu, disp, motion, prev)
            prev = (start, end, idx_gpu)
        
        # process the last `prev` block
        self._update_motion_by_prev(points_gpu, disp, motion, prev)
        
        return motion.float()

    def inverse_forward(self):
        pass
        
        
    
def rbf_motion_kdtree(
    points: np.ndarray,     #(N, 3)
    ctrl_pts: np.ndarray,   #(M, 3)
    ctrl_disp: np.ndarray,  #(M, 3)
    *,
    sigma: float = 30.0,
    chunk_size: int = 1_000_000,
    radius_factor: float = 3.0,
    k: int = 64,
    device: torch.device|None =None,
):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    radius = sigma * radius_factor
    N = points.shape[0]
    
    points_gpu = torch.from_numpy(points).pin_memory().to(device, torch.half, non_blocking=True)
    ctrl_pts_gpu = torch.from_numpy(ctrl_pts).pin_memory().to(device, torch.half, non_blocking=True)
    ctrl_disp_gpu = torch.from_numpy(ctrl_disp).pin_memory().to(device, torch.half, non_blocking=True)
    motion = torch.zeros((N, 3), dtype=torch.half, device=device)
    tree_n_gpu = torch.tensor(ctrl_pts.shape[0], dtype=torch.half, device=device)

    tree = KDTree(ctrl_pts)

    copy_stream = torch.cuda.Stream(device=device)

    prev :tuple[int, int, torch.Tensor]|None = None
    
    def _update_motion_by_prev():
        if prev is None:
            return
        
        prev_start, prev_end, prev_idx = prev

        # waiting for copy
        torch.cuda.current_stream().wait_stream(copy_stream)

        # for points with neighbors less than k, KDTree will fill its dist by inf 
        # and set idx by tree.n (tree.n == ctrl_pts.shape[0] == N)
        mask = prev_idx != tree_n_gpu
        prev_idx[~mask] = 0

        ctrl_pts_t = ctrl_pts_gpu[prev_idx]
        ctrl_disp_t = ctrl_disp_gpu[prev_idx]
        points_t = points_gpu[prev_start:prev_end]

        # gaussian kernel
        diff = points_t.unsqueeze(1) - ctrl_pts_t
        r2 = (diff ** 2).sum(-1)
        w = torch.exp(-r2 / (2 * sigma**2))
        # we used 0 to fill invalid idx before, i.e. calculate `w` with points_t[0], need to be removed.
        w = w * mask
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)

        chunk_motion = (w.unsqueeze(-1) * ctrl_disp_t).sum(1)
        motion[prev_start:prev_end] = chunk_motion


    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        
        # ---------- CPU KDTree ----------
        # set distance_upper_bound=radius thus points with dist > radius will not return
        # set workers = -1 to use all CPU
        _, idx = tree.query(points[start:end], k=k, distance_upper_bound=radius, workers=-1)

        idx_cpu = torch.from_numpy(idx)

        # ---------- async H2D ----------
        with torch.cuda.stream(copy_stream):
            idx_gpu = idx_cpu.to(device, non_blocking=True)

        _update_motion_by_prev()

        prev = (start, end, idx_gpu)
    
    _update_motion_by_prev()     # process the last `prev` block
    return motion.cpu().float().numpy()

# %% [markdown]
# 测试生成更加连续光滑的运动

# %%
new_phase_num = 50

rbf = KDTreeRBF(landmark_surface).to(torch.device('cuda')).eval()

new_deformed_cavities:list[pv.PolyData] = []
rbf.cache_points(landmark_surface.points.copy())
for phase_j in range(new_phase_num):
    new_mesh = landmark_surface.copy()
    disp = rbf.forward(CardiacPhase.from_index(phase_j, new_phase_num)).cpu().numpy()
    new_mesh.points += disp
    new_deformed_cavities.append(new_mesh)

coronary_mesh_list: list[pv.PolyData] = []
rbf.cache_points(coronary_mesh.points)
for phase_j in range(new_phase_num):
    new_mesh = coronary_mesh.copy()
    disp = rbf.forward(CardiacPhase.from_index(phase_j, new_phase_num)).cpu().numpy()
    new_mesh.points += disp
    coronary_mesh_list.append(new_mesh)


# %%
plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.camera_position = pv.CameraPosition(
    (-68.16251727594782, 104.066792868929, 344.7257297397248),
    label_surface.center,
    (0.7523987560658962, -0.5745229979148703, 0.3222102368600382)
)
plotter.open_gif(str(temp_output / "moving.gif"), fps=30)
for cavity, coronary in zip(new_deformed_cavities, coronary_mesh_list):
    plotter.clear()
    plotter.add_mesh(cavity, scalars="label", opacity=0.5)
    plotter.add_mesh(coronary, color="red", opacity=0.8)
    plotter.write_frame()
plotter.show()

# %% [markdown]
# ## 使用 RBF 生成密集形变场-1

# %%
from generate_4d_heart.roi import ROI

# 根据 心腔标签范围确定 ROI 并进行裁剪和缩放以调整到合适的尺寸供后续 4D ddf 生成使用
# ori_img -- crop --> img_1(shape_1, affine_1) -- zoom --> img_2(shape_2, affine_2)
shape_2 = (144, 144, 128)
roi = ROI.get_from_cavity(cavity_img, target_shape=shape_2, padding=20)
coronary_1 = roi.crop(coronary_img)
cavity_1 = roi.crop(cavity_img)
shape_1 = cavity_1.shape
affine_1 = cavity_1.affine
assert affine_1 is not None
nib.save(cavity_1, temp_output / "cavity_1.nii.gz")
nib.save(coronary_1, temp_output / "coronary_1.nii.gz")

cavity_2 = roi.crop_zoom(cavity_img, is_label=True)
affine_2 = cavity_2.affine
assert affine_2 is not None

grid_2 = np.meshgrid(
    np.arange(shape_2[0]),
    np.arange(shape_2[1]),
    np.arange(shape_2[2]),
    indexing='ij'
)

coords_2 = np.stack(grid_2, axis=-1)  # (144, 144, 128, 3)
coords_2 = apply_affine(coords_2, affine_2)  # voxel to world


# %%
from torch.nn import functional as F

torch.cuda.empty_cache()

device = torch.device('cuda:1')
rbf = KDTreeRBF(landmark_surface).to(device).eval()
rbf.cache_points(coords_2.reshape(-1, 3))

dx_world_2 = rbf.forward(CardiacPhase.from_index(0,20))
dx_world_2 = dx_world_2.reshape(shape_2 + (3,)) # (144, 144, 128, 3)
dx_world_1 = F.interpolate(
    dx_world_2.permute(3, 0, 1, 2).unsqueeze(0), # (1, 3, 144, 144, 128)
    size=shape_1,
    mode='trilinear',
)   # (1, 3, *shape_1)
dx_world_1 = dx_world_1.squeeze(0).permute(1, 2, 3, 0)   # (shape_1, 3)

assert affine_1 is not None
R_voxel_to_world = torch.tensor(affine_1[:3, :3]).float().to(device)
R_world_to_voxel: torch.Tensor = torch.linalg.inv(R_voxel_to_world)
dx_coord_1 = dx_world_1 @ R_world_to_voxel.T


# %%
def save_vector_nii(dx_world: torch.Tensor, affine: np.ndarray|None, path: Path):
    # 这里存储的是物理位移，否则会有方向问题
    # 可以在导入3Dslicer后手动选择invert
    dvf_np = dx_world.detach().cpu().numpy().astype(np.float32)
    
    dvf_nii = nib.Nifti1Image(dvf_np, affine)

    dvf_nii.header.set_intent('vector', (), '') 
    nib.save(dvf_nii, path)

save_vector_nii(dx_world_1, affine_1, temp_output / "dx_world_1.nii.gz")

# %%
def invert_displacement_field(
    u_forward: torch.Tensor, 
    n_iters: int=10
) -> torch.Tensor:
    """
    使用不动点迭代法求解反向位移场 v, 使得对于每个体素坐标 y, 有 v(y) = -u(y + v(y))。
    u_forward: 正向位移场 Tensor of shape (W, H, D, 3), 单位是体素
    n_iters: 迭代次数
    """
    # 1. 维度准备: (W, H, D, 3) -> (1, 3, D, H, W)
    # 注意：根据你的坐标习惯，可能需要调换最后一位的顺序 (x,y,z)
    u = u_forward.permute(3, 2, 1, 0).unsqueeze(0) 
    device = u.device
    C, D, H, W = u.shape[1:]
    
    # 2. 生成标准的采样网格 [-1, 1]
    grid_d, grid_h, grid_w = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    identity_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0) # (1, D, H, W, 3)

    # 3. 如果你的 u 是体素单位，需要转成 [-1, 1] 相对单位
    # 如果已经是相对单位则跳过此步
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

    # 5. 转回原始 shape (W, H, D, 3)
    v_final = v.squeeze(0).permute(3, 2, 1, 0)
    
    # 如果之前缩放了单位，这里记得乘回来
    v_final[..., 0] *= (W / 2.0)
    v_final[..., 1] *= (H / 2.0)
    v_final[..., 2] *= (D / 2.0)
    
    return v_final

inv_dx_coord_1 = invert_displacement_field(dx_coord_1)

# %% [markdown]
# 验证 forward 和 inverse 是否近似互逆

# %%
x = torch.randint(high = min(*dx_coord_1.shape[:3]), size=(6, 3)).to(dx_coord_1.device)
u = dx_coord_1[x[:,0], x[:,1], x[:,2]]
y = x + u
y_int = y.to(torch.int)
v = inv_dx_coord_1[y_int[:,0], y_int[:,1], y_int[:,2]]
x_new = y_int + v
print(f"{x-x_new=}")

# %% [markdown]
# 验证 ddf 正确性

# %%
from monai.networks.blocks.warp import Warp
device = torch.device('cuda:1')
warp_label = Warp(mode="nearest", padding_mode="zeros").to(device)
cavity_warpped_1 = warp_label(
    torch.from_numpy(cavity_1.get_fdata()).squeeze()[None, None].float().to(device),
    inv_dx_coord_1.permute(3, 0, 1, 2).unsqueeze(0).to(device)
)
nib.save(nib.Nifti1Image(cavity_warpped_1.squeeze().cpu().numpy(), affine_1), temp_output / "cavity_warpped_1.nii.gz")

# %% [markdown]
# ## 使用 RBF 生成密集形变场-2
# 
# 上面的方法首先对心腔区域进行了裁剪并缩放到指定shape，这是为了降低计算复杂度。同时增加对不同尺度模型的适应性。
# 
# 从基本原理上说，上面的过程与下面类似

# %%
cavity_img, cavity_data = read_nii(cavity_path)
coronary_img, coronary_data = read_nii(coronary_path)

shape = cavity_img.shape
affine = cavity_img.affine
assert affine is not None

grid = np.meshgrid(
    np.arange(shape[0]),
    np.arange(shape[1]),
    np.arange(shape[2]),
    indexing='ij'
)

coords = np.stack(grid, axis=-1)  # (144, 144, 128, 3)
coords = apply_affine(coords, affine)  # voxel to world

device = torch.device('cuda:0')
rbf = KDTreeRBF(landmark_surface).to(device).eval()
rbf.cache_points(coords.reshape(-1, 3))
dx_world: torch.Tensor = rbf.forward(CardiacPhase.from_index(0, 20))
dx_world = dx_world.reshape(shape + (3,))
# 到上面为止都是正确的

affine_inv = np.linalg.inv(affine)
R_world_to_voxel = affine_inv[:3, :3]

dx_coord = dx_world @ torch.from_numpy(R_world_to_voxel.T)
save_vector_nii(dx_world, affine, temp_output / "dx_world.nii.gz")

# %%
R_world_to_voxel

# %%
inv_dx_coord = invert_displacement_field(dx_coord)

# %% [markdown]
# 验证 生成了正确的位移（世界坐标）

# %%
rbf = KDTreeRBF(landmark_surface).to(device).eval()

crnr_pts = coronary_mesh.points
disp = rbf.forward(CardiacPhase.from_index(0, 20), crnr_pts)
crnr_pts_voxel = apply_affine(crnr_pts, affine_inv)
crnr_pts[:3], crnr_pts_voxel[:3].round(), disp[:3]



# %%
coords[275, 254, 33], dx_world[275, 254, 33]

# %% [markdown]
# 验证voxel坐标的反向位移（DDF）正确性

# %%
from monai.networks.blocks.warp import Warp
device = torch.device('cuda:1')
warp_label = Warp(mode="nearest", padding_mode="zeros").to(device)
cavity_warpped = warp_label(
    torch.from_numpy(cavity_data).squeeze()[None, None].float().to(device),
    inv_dx_coord.permute(3, 0, 1, 2).unsqueeze(0).to(device)
)

# %%
nib.save(nib.Nifti1Image(cavity_warpped.squeeze().cpu().numpy(), affine), temp_output / "cavity_warpped.nii.gz")


