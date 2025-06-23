from pathlib import Path

import numpy as np
from scipy import ndimage
import pyvista as pv
import pyacvd
import torchcpd
from nibabel.nifti1 import Nifti1Image
import nibabel as nib

from . import SSM_DIRECTION, NUM_TOTAL_CAVITY_LABEL, NUM_TOTAL_PHASE, NUM_TOTAL_POINTS
from .utils import get_maybe_flip_transform

def _get_largest_connected_component(data: np.ndarray) -> np.ndarray:
    """
    Obtain the largest connected region in the binary image. Copy from Phantom/script/get_surface_cloud.py
    Args:
        data: the binary image
    Returns:
        np.ndarray: the largest connected region in the binary image
    """
    res = ndimage.label(data)
    assert isinstance(res, tuple) and isinstance(res[0], np.ndarray) and isinstance(res[1], int)
    labeled_data, num_features = res
    assert num_features > 0
    sizes = ndimage.sum(data, labeled_data, range(num_features + 1))
    largest_component = sizes.argmax()
    return (labeled_data == largest_component).astype(np.uint8)

def get_surface_from_label(
        label: np.ndarray,
        affine: np.ndarray
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
    label = get_maybe_flip_transform(affine)(label)
    
    label_ids = sorted(np.unique(label).astype(np.int8))[1:]
    surface_all = pv.PolyData()
    
    for label_id in label_ids:
        assert label_id > 0
        if label_id == 1:
            # The myocardial part of the left ventricle is combined with the left ventricular cavity part
            mask_1 = (label == 1).astype(np.uint8)
            mask_2 = (label == 2).astype(np.uint8)
            mask = mask_1 + mask_2
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = (label == label_id).astype(np.uint8)
        
        structure = ndimage.generate_binary_structure(3, 1)
        mask = _get_largest_connected_component(mask)
        mask = ndimage.binary_closing(mask, iterations=1, structure=structure)
        mask = ndimage.binary_opening(mask, iterations=1, structure=structure)

        surface = pv.wrap(mask).contour([1], method="flying_edges").triangulate().smooth_taubin(n_iter=50).clean()
        cluster = pyacvd.Clustering(surface)
        cluster.subdivide(2)
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
    moving_labels = source_surface.point_data["label"]
    fix_labels = target_surface.point_data["label"]
    labels = np.unique(moving_labels)
    labels = labels[labels != 0]
    new_cloud = pv.PolyData()
    new_points_all, _ = torchcpd.RigidRegistration(X=target_surface.points, Y=source_surface.points, device=device).register()
    new_points_all, _ = torchcpd.AffineRegistration(X=target_surface.points, Y=new_points_all.cpu().numpy(), device=device).register()
    for label in labels:
        source_points = new_points_all[moving_labels == label]
        target_points = target_surface.points[fix_labels == label]
        new_points, _ = torchcpd.AffineRegistration(X=target_points, Y=source_points.cpu().numpy(), device=device).register()
        new_points, _ = torchcpd.DeformableRegistration(X=target_points, Y=new_points.cpu().numpy(), device=device, kwargs=deform_kwargs).register()
        cloud = pv.PolyData(new_points.cpu().numpy())
        cloud.point_data["label"] = np.ones(cloud.n_points).astype(np.uint8) * label
        new_cloud = new_cloud.merge(cloud)
    
    res = source_surface.copy()
    res.points = new_cloud.points
    return res

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
    return polydata.extract_cells(face_indices).extract_surface()

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
            template_surface: pv.PolyData | Path,
            b_motion: np.ndarray | Path,
            P_motion: np.ndarray | Path, 
    ):
        """
        Args:
            template_surface: pv.PolyData, the template surface
            b_motion: np.ndarray, the dense deformation vector fields for each label,  shape = (num_labels(L), num_phases(N_j), num_components(N_m)),
            P_motion: np.ndarray, the dense displacement fields for each label, shape = (num_labels(L), num_components(N_m), num_points(M), 3)
        """
        if isinstance(template_surface, Path):
            template_surface = pv.read(template_surface)
        if isinstance(b_motion, Path):
            b_motion = np.load(b_motion)
        if isinstance(P_motion, Path):
            P_motion = np.load(P_motion)
        assert isinstance(template_surface, pv.PolyData) and isinstance(b_motion, np.ndarray) and isinstance(P_motion, np.ndarray)
        num_labels, _, num_points, num_dim = P_motion.shape
        num_labels_, num_phases = b_motion.shape[:2]
        assert num_labels == num_labels_ == NUM_TOTAL_CAVITY_LABEL and num_phases == NUM_TOTAL_PHASE and num_dim == 3 and NUM_TOTAL_POINTS == num_points

        self.template_surface = template_surface
        self.b_motion = b_motion
        self.P_motion = P_motion
    
    def apply(
            self, 
            label: Nifti1Image,
            device: int,
            num_components_used: int = 1,
    ) -> 'SSM_Result':
        """
        Apply the SSM to the label to generate 4d cavity label.
        Args:
            label: Nifti1Image, the label to apply SSM
            device: the device to use
            num_components_used: the number of PCA components used to deform motion
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
        zooming_rate = float(np.mean(landmark_size / template_size))
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
            label_i: landmark_surface.points[landmark_surface.point_data['label'] == label_i+1] for label_i in range(NUM_TOTAL_CAVITY_LABEL)
        }

        for points in points_dict.values():
            if len(points) != NUM_TOTAL_POINTS:
                raise ValueError(f"Number of points should be {NUM_TOTAL_POINTS}")

        res = []
        for phase in range(NUM_TOTAL_PHASE):
            all_deformed_points = pv.PolyData()
            for label_i in range(NUM_TOTAL_CAVITY_LABEL):
                b = self.b_motion[label_i, phase, :num_components_used] * motion_zoom_rate
                P = self.P_motion[label_i, :num_components_used]
                deformation = np.einsum('K, KMD -> MD', b, P)
                deformed_points = pv.PolyData(points_dict[label_i] + deformation)
                deformed_points.point_data['label'] = np.ones(deformed_points.n_points, dtype=np.uint8)* (label_i+1)
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
        self.flips = get_maybe_flip_transform(self.original_label.affine)
    
    def get_motion_volume(self, phase: int) -> Nifti1Image:
        """
        Args:
            phase: int, the phase to get motion volume, start from 0 and less than NUM_TOTAL_PHASE.
        Returns:
            Nifti1Image: the motion label volume at the specified phase.
        """
        assert 0 <= phase < NUM_TOTAL_PHASE
        cavity_label = self.cavity_surfaces[phase]
        motion_label = polydata_to_label_volume(cavity_label, self.original_label.shape)
        return Nifti1Image(self.flips(motion_label), affine=self.original_label.affine)
    
    def get_landmark_volume(self):
        landmark_volume = polydata_to_label_volume(self.landmark_vtk, self.original_label.shape)
        return Nifti1Image(self.flips(landmark_volume), affine=self.original_label.affine)

    def save_gif(self, save_path: Path):
        """
        Args:
            save_path: Path, the path to save the gif
        """
        plotter = pv.Plotter(off_screen=True)
        plotter.open_gif(str(save_path))
        for polydata in self.cavity_surfaces:
            plotter.clear()
            plotter.add_mesh(polydata, scalars="label", opacity=0.5)
            plotter.write_frame()
        plotter.close()


