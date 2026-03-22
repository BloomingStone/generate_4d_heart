from pathlib import Path

import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
import pyvista as pv
import pyacvd
import torchcpd
from nibabel.nifti1 import Nifti1Image
import nibabel as nib
from scipy.spatial import KDTree
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_edge_loss
import logging

from generate_4d_heart import NUM_TOTAL_POINTS, CavityLabel, NUM_TOTAL_PHASE
from .utils import MaybeFlipTransform

logger = logging.getLogger(__name__)

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
    sizes = ndimage.sum(data_cp, labeled_data, cp.arange(num_features + 1))  # type: ignore
    largest_component = sizes.argmax()
    return (labeled_data == largest_component).astype(cp.uint8)

def identify_faces(polydata: pv.PolyData) -> pv.PolyData:
    """
    Identify faces in the polydata and assign labels to them.
    Args:
        polydata (pv.PolyData): Input polydata with point data "label".
    Returns:
        pv.PolyData: Polydata with face labels.
    """
    labels = polydata.point_data["label"]
    faces = polydata.faces.reshape(-1, 4)[:, 1:]
    face_labels = np.array([
        labels[faces[i, 0]] if labels[faces[i, 0]] == labels[faces[i, 1]] == labels[faces[i, 2]] else 0
        for i in range(faces.shape[0])
    ])
    res= polydata.copy()
    res.cell_data["label"] = face_labels
    return res

def get_surface_from_label(
        label: np.ndarray,
        affine: np.ndarray | None
) -> pv.PolyData:
    """
    Get the cloud from the label. Copy from Phantom/script/get_surface_cloud.py
    Args:
        nii_path: the path to the nii file
        max_point_num: the maximum number of points in the cloud for each label
        direction: the 
    Returns:
        pv.PolyData: the cloud
        np.ndarray: the affine matrix of the nii file
    """
    label = MaybeFlipTransform(affine)(label, is_vector_field=False)    # TODO 这里更正确的做法是使用affine将voxel坐标转换到world坐标系
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
    return surface_all


def pytorch3d_refine(
    source_pv: pv.PolyData, 
    target_pv: pv.PolyData, 
    device: str = "cuda:0",
    steps: int = 100,
    lr: float = 0.01,
    w_chamfer: float = 1.0,
    w_laplacian: float = 0.1,
    w_edge: float = 0.1
) -> pv.PolyData:
    """
    使用 PyTorch3D 对已经初步对齐的网格进行高精度形变微调。
    """
    # PyVista -> PyTorch3D 
    src_verts = torch.from_numpy(source_pv.points).float().to(device)
    src_faces = torch.from_numpy(np.array(source_pv.faces).reshape(-1, 4)[:, 1:]).long().to(device) #faces[i] = [n, id1, id2, id3]
    tgt_verts = torch.from_numpy(target_pv.points).float().to(device)

    src_mesh = Meshes(verts=[src_verts], faces=[src_faces])
    
    # optimize offsets of verts for stable topol structure
    deform_verts = torch.full(src_verts.shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()

        new_mesh = src_mesh.offset_verts(deform_verts)

        # A. Chamfer Loss: Bring the point cloud closer to the target surface
        loss_chamfer, _ = chamfer_distance(new_mesh.verts_packed().unsqueeze(0),  #type: ignore
                                           tgt_verts.unsqueeze(0))
        
        # B. Laplacian Loss: keep smooth
        loss_laplacian = mesh_laplacian_smoothing(new_mesh, method="uniform")
        
        # C. Edge Loss: prevent the stretching distortion of the triangle
        loss_edge = mesh_edge_loss(new_mesh)
        
        total_loss = w_chamfer * loss_chamfer + w_laplacian * loss_laplacian + w_edge * loss_edge  #type: ignore
        if step % 10 == 0:
            logger.debug(f"{step=:03d}, {loss_chamfer=:.5f} {loss_laplacian=:.5f} {loss_edge=:.5f}")
        
        total_loss.backward()
        optimizer.step()

    final_verts = src_mesh.offset_verts(deform_verts).verts_packed().detach().cpu().numpy()  #type: ignore
    refined_pv = source_pv.copy()
    refined_pv.points = final_verts
    return refined_pv

def deform_surface(
        source_surface: pv.PolyData, 
        target_surface: pv.PolyData, 
        device: int,
        **deform_kwargs
    ) -> pv.PolyData:
    """
    Use deformable registration to deform the source surface to the target surface. Copy from Phantom/script/align_surface.py
    Args:
        source_surface: the source surface
        target_surface: the target surface
        device: the device to use
        deform_kwargs: the kwargs for deformable registration
    Returns:
        pv.PolyData: the deformed surface
    """
    source_surface = source_surface.copy()
    moving_labels = source_surface.cell_data["label"]
    fix_labels = target_surface.cell_data["label"]
    labels = np.unique(moving_labels)
    labels = labels[labels != 0]
    new_cloud = pv.PolyData()
    new_points_all, _ = torchcpd.RigidRegistration(X=target_surface.points, Y=source_surface.points, device=device).register()
    new_points_all, _ = torchcpd.AffineRegistration(X=target_surface.points, Y=new_points_all.cpu().numpy(), device=device).register()
    
    source_surface.points = new_points_all.cpu().numpy()
    for label in sorted(labels):
        logger.debug(f"process label:{label+1}")
        source_submesh: pv.PolyData = source_surface.extract_cells(moving_labels == label).extract_surface(algorithm=None)
        target_submesh: pv.PolyData = target_surface.extract_cells(fix_labels == label).extract_surface(algorithm=None)
        new_points, _ = torchcpd.AffineRegistration(X=target_submesh.points, Y=source_submesh.points, device=device).register()
        new_points, _ = torchcpd.DeformableRegistration(X=target_submesh.points, Y=new_points.cpu().numpy(), device=device, kwargs=deform_kwargs).register()
        source_submesh.points = new_points.cpu().numpy()
        
        source_submesh = pytorch3d_refine(
            source_pv=source_submesh, 
            target_pv=target_submesh, 
            device=f"cuda:{device}",
            steps=200,
        )
        
        new_cloud = new_cloud.merge(source_submesh)
    
    return new_cloud

def _extract_faces_by_label(polydata: pv.PolyData, label: int) -> pv.PolyData:
    """
    Extract faces from the polydata based on the specified label.
    Args:
        polydata (pv.PolyData): Input polydata with cell data "label".
        label (int): The label to extract.
    Returns:
        pv.PolyData: Extracted faces with the specified label.
    """
    labels = polydata.cell_data["label"]
    unique_labels = np.unique(labels)
    if label not in unique_labels:
        raise ValueError(f"Label {label} not found in polydata.")
    face_indices = np.where(labels == label)[0]
    return polydata.extract_cells(face_indices).extract_surface(algorithm=None)

def polydata_to_label_volume(
    polydata: pv.PolyData,
    output_shape: tuple[int, ...], 
) -> np.ndarray :
    """
    将 polydata surface（以 label 区分）转换为 label segmentation volumes。

    Args:
        polydata: pv.PolyData, 输入的 polydata surface。
        output_shape: tuple[int, ...], 体素坐标系的形状。
    Returns:
        label_volume: np.ndarray, 体素坐标系的标签体积。
    """
    reference_volume = pv.ImageData(dimensions=output_shape)
    label_volume = np.zeros(output_shape, dtype=np.uint8)
    labels = np.unique(polydata.point_data["label"]).tolist()
    labels = [int(label) for label in labels if label > 0 and label != 2]
    labels = sorted(labels, reverse=True)
    labels.append(2)   # 左心室部分 label=2， 因为此前和左心肌合并处理，故此处需要最后处理，以覆盖在所有label之上

    for label in labels:
        face_of_label = _extract_faces_by_label(polydata, label)
        mask_image_data = face_of_label.voxelize_binary_mask(reference_volume = reference_volume)
        mask = mask_image_data.point_data['mask'].reshape(output_shape, order='F')
        label_volume[mask == 1] = label
    
    return label_volume


class SSM:
    def __init__(
            self,
            template_surface: pv.DataObject | Path,
            b_motion: np.ndarray | Path,
            P_motion: np.ndarray | Path, 
    ):
        """
        Args:
            template_surface: pv.PolyData, the template surface
            b_motion: np.ndarray, the dense deformation vector fields for each label,  shape = (num_labels(L), num_phases(N_j), num_components(N_m)),
            P_motion: np.ndarray, the dense displacement fields for each label, shape = (num_labels(L), num_components(N_m), num_points(M), 3)
        """
        temp_path = None
        if isinstance(template_surface, Path):
            temp_path = template_surface
            template_surface_ = pv.read(template_surface)
        else:
            template_surface_ = template_surface
        if isinstance(b_motion, Path):
            b_motion = np.load(b_motion)
        if isinstance(P_motion, Path):
            P_motion = np.load(P_motion)
        assert isinstance(template_surface_, pv.PolyData) and isinstance(b_motion, np.ndarray) and isinstance(P_motion, np.ndarray)
        num_labels, _, num_points, num_dim = P_motion.shape
        num_labels_, num_phases = b_motion.shape[:2]
        assert num_labels == num_labels_ == len(CavityLabel) and num_phases == NUM_TOTAL_PHASE and num_dim == 3 and NUM_TOTAL_POINTS == num_points

        if 'label' not in template_surface_.cell_data:
            if 'label' not in template_surface_.point_data:
                raise ValueError("The template surface should have cell data 'label'.")
            template_surface_ = identify_faces(template_surface_)
            if temp_path is not None:
                template_surface_.save(temp_path)
        
        self.template_surface = template_surface_
        self.b_motion = b_motion
        self.P_motion = P_motion
    
    def apply(
            self, 
            label: Nifti1Image,
            device: int,
            moving_enhance_factor: float,
            num_components_used: int = 1,
    ) -> 'SSM_Result':
        """
        Apply the SSM to the label to generate 4d cavity label.
        Args:
            label: Nifti1Image, the label to apply SSM
            device: the device to use
            num_components_used: the number of PCA components used to deform motion,
            moving_enhance_factor: the factor to enhance the motion
        Returns:
            SSM_Result: the result of applying SSM
        """
        affine = label.affine
        assert affine is not None
        label_surface = get_surface_from_label(label.get_fdata(), affine)
        landmark_surface = deform_surface(self.template_surface, label_surface, device)
        template_bounding_box = self.template_surface.bounds
        landmark_bounding_box = landmark_surface.bounds
        template_size = np.array([template_bounding_box[2*i+1] - template_bounding_box[2*i] for i in range(3)])
        landmark_size = np.array([landmark_bounding_box[2*i+1] - landmark_bounding_box[2*i] for i in range(3)])
        zooming_rate = float(np.mean(landmark_size / template_size)) * moving_enhance_factor
        cavity_labels = self._generate_4d_cavity(landmark_surface, zooming_rate, num_components_used)
        return SSM_Result(label, landmark_surface, cavity_labels)


    def _generate_4d_cavity(
            self,
            landmark_surface: pv.PolyData,
            motion_zoom_rate: float,
            num_components_used: int
    ) -> list[pv.PolyData]:
        """
        Generate 4d cavity label from the landmark surface and the template cloud. Copy from Phantom/script/generate_4d_image.py
        Args:
            landmark_surface: the landmark surface
            motion_zoom_rate: the zoom rate of the motion,
            num_components_used: the number of PCA components used to deform motion
        Returns:
            list[pv.PolyData]: the 4d cavity label for each phase
        """
        points_dict = {
            label_i: landmark_surface.points[landmark_surface.point_data['label'] == label_i] for label_i in CavityLabel
        }

        for points in points_dict.values():
            if len(points) != NUM_TOTAL_POINTS:
                raise ValueError(f"Number of points should be {NUM_TOTAL_POINTS}")

        res = []
        for phase in range(NUM_TOTAL_PHASE):
            all_deformed_points = pv.PolyData()
            for label in CavityLabel:
                label_i = label-1
                b = self.b_motion[label_i, phase, :num_components_used] * motion_zoom_rate
                P = self.P_motion[label_i, :num_components_used]
                deformation = np.einsum('K, KMD -> MD', b, P)
                deformed_points = pv.PolyData(points_dict[label] + deformation)
                deformed_points.point_data['label'] = np.ones(deformed_points.n_points, dtype=np.uint8)* (label+1)
                all_deformed_points = all_deformed_points.merge(deformed_points)
            assert isinstance(all_deformed_points, pv.PolyData)
            t = landmark_surface.copy()
            t.points = all_deformed_points.points
            res.append(t)
        return res


class SSM_Result:
    def __init__(
            self,
            original_cavity_label: Nifti1Image,
            landmark_vtk: pv.PolyData,
            cavity_surfaces: list[pv.PolyData],
    ):
        self.original_label = original_cavity_label
        self.landmark_vtk = landmark_vtk
        self.cavity_surfaces = cavity_surfaces
        assert self.original_label.affine is not None
        self.flips = MaybeFlipTransform(self.original_label.affine)
    
    def get_motion_volume(self, phase: int) -> Nifti1Image:
        """
        Get cavity at given phase index
        Args:
            phase: int, the phase to get motion volume, start from 0 and less than NUM_TOTAL_PHASE.
        Returns:
            Nifti1Image: the motion label volume at the specified phase.
        """
        assert 0 <= phase < NUM_TOTAL_PHASE
        cavity_label = self.cavity_surfaces[phase]
        motion_label = polydata_to_label_volume(cavity_label, self.original_label.shape)
        return Nifti1Image(self.flips(motion_label), affine=self.original_label.affine)
    
    def get_landmark_volume(self) -> Nifti1Image:
        landmark_volume = polydata_to_label_volume(self.landmark_vtk, self.original_label.shape)
        return Nifti1Image(self.flips(landmark_volume), affine=self.original_label.affine)

    def save_gif(self, save_path: Path):
        """
        Args:
            save_path: Path, the path to save the gif
        """
        try:
            plotter = pv.Plotter(off_screen=True)
            plotter.open_gif(str(save_path))
            for polydata in self.cavity_surfaces:
                plotter.clear()
                plotter.add_mesh(polydata, scalars="label", opacity=0.5)
                plotter.write_frame()
            plotter.close()
        except Exception as e:
            print(e)


