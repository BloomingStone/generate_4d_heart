import cupy as cp
from cupyx.scipy import ndimage
import numpy as np
import pyvista as pv
import pyacvd
from nibabel import Nifti1Image

from generate_4d_heart import CavityLabel, NUM_TOTAL_POINTS


def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply affine transformation to points with shape (..., 3)."""
    assert points.shape[-1] == 3, "The last dimension of points must be 3."
    assert affine.shape == (4, 4), "affine must be shape (4, 4)."
    return points @ affine[:3, :3].T + affine[:3, 3]


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

def label_to_polydata(
        label: np.ndarray,
        affine: np.ndarray
) -> pv.PolyData:
    """
    Get the cloud from the label. Copy from Phantom/script/get_surface_cloud.py
    Args:
        label: the label array
        affine: the affine matrix of the nii file
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
            mask[label_cp == int(CavityLabel.LV_MYO)] = 1
            mask[label_cp == int(CavityLabel.LV)] = 1
            mask = (mask > 0).astype(cp.uint8)
        else:
            mask = (label_cp == int(label_id)).astype(cp.uint8)
        
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


def label_nii_to_polydata(
    label_nii: Nifti1Image
) -> pv.PolyData:
    """
    Convert label nii to polydata. Copy from Phantom/script/get_surface_cloud.py
    Args:
        label_nii: the label nii file
    Returns:
        pv.PolyData: the polydata
    """
    label = label_nii.get_fdata().astype(np.uint8)
    affine = label_nii.affine
    assert affine is not None
    return label_to_polydata(label, affine)

def _extract_faces_by_label(polydata: pv.PolyData, label: CavityLabel) -> pv.PolyData:
    """
    Extract faces from the polydata based on the specified label.
    Args:
        polydata (pv.PolyData): Input polydata with cell data "label".
        label (CavityLabel): The label to extract.
    Returns:
        pv.PolyData: Extracted faces with the specified label.
    """
    labels = polydata.cell_data["label"]
    unique_labels = np.unique(labels)
    if label not in unique_labels:
        raise ValueError(f"Label {label} not found in polydata.")
    face_indices = np.where(labels == label)[0]
    return polydata.extract_cells(face_indices).extract_surface(algorithm=None)

def polydata_to_label(
    polydata: pv.PolyData,
    ref_shape: tuple[int, ...],
    ref_affine: np.ndarray|None
) -> np.ndarray:
    if ref_affine is None:
        affine_inv = np.eye(4)
    else:
        affine_inv = np.linalg.inv(ref_affine)
    
    polydata = polydata.copy()
    polydata.points = apply_affine(polydata.points, affine_inv)
    
    reference_volume = pv.ImageData(dimensions=ref_shape)
    
    label_volume = np.zeros(ref_shape, dtype=np.uint8)
    polydata_labels: list[int] = np.unique(polydata.point_data["label"]).tolist()
    labels: list[CavityLabel] = []
    for label in reversed(CavityLabel):
        assert label in polydata_labels
        if label == CavityLabel.LV:
            continue
        labels.append(label)
    labels.append(CavityLabel(2))   # 左心室部分 label=2， 因为此前和左心肌合并处理，故此处需要最后处理，以覆盖在所有label之上

    for label in labels:
        face_of_label = _extract_faces_by_label(polydata, label)
        mask_image_data = face_of_label.voxelize_binary_mask(reference_volume = reference_volume)
        mask = mask_image_data.point_data['mask'].reshape(ref_shape, order='F')
        label_volume[mask == 1] = label
    
    return label_volume

def polydata_to_label_nii(
    polydata: pv.PolyData,
    ref_nii: Nifti1Image,
) -> Nifti1Image:
    """
    将以 cell_data["label"] 区分的 polydata surface 转换为 label segmentation volumes。
    """
    shape = ref_nii.shape
    affine = ref_nii.affine
    label_volume = polydata_to_label(polydata, shape, affine)
    
    return Nifti1Image(label_volume, affine)
