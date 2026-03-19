# NEW IMPLEMENT OF SSM
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pyvista as pv
import nibabel as nib
import torch
import torchcpd
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_edge_loss
import einops

from generate_4d_heart.ssm import polydata_io
from generate_4d_heart import CavityLabel


logger = logging.getLogger(__name__)
SSM_DATA_DIR = Path(__file__).parent / "ssm_world"

@dataclass
class SSMReader:
    def __init__(
        self,
        ssm_dir: Path=SSM_DATA_DIR
    ):
        self.ssm_dir = ssm_dir
        self.ssm_template: pv.PolyData = pv.read(ssm_dir / "ssm_template.vtk")  #type: ignore
        assert isinstance(self.ssm_template, pv.PolyData)
        if  "label" not in self.ssm_template.cell_data:
            raise ValueError("The template surface should have label in cell data or point data")
        self.b: np.ndarray = np.load(ssm_dir / "b_motion.npy")
        self.P: np.ndarray = np.load(ssm_dir / "P_motion.npy")
        
                
        self.n_labels, self.n_phases, self.n_components = self.b.shape
        n_labels_, n_components_, self.n_points, n_dim = self.P.shape
        assert self.n_labels == n_labels_ == len(CavityLabel) 
        assert self.n_components == n_components_ 
        assert n_dim == 3


    def load(
        self, 
        cavity: nib.Nifti1Image|Path|np.ndarray,
        affine: np.ndarray|None = None,
        num_components_used: int = 1,
        motion_multiplier: float = 1.0
    ) -> "SSMResult":
        """
        Load SSM data for the given cavity and affine transformation.
        Args:
            cavity: The cavity surface, can be a Nifti1Image, a file path to a Nifti image, 
            or a numpy array of shape (H, W, D).
            affine: The affine transformation matrix of the cavity. Required if cavity is a 
            numpy array.
            num_components_used: The number of principal components to use for deformation. 
            Default is 1.
            motion_multiplier: A multiplier for the motion deformation. Default is 1.0.
        """
        match cavity, affine:
            case nib.Nifti1Image(), None:
                surface = polydata_io.label_nii_to_polydata(cavity)
            case Path(), None:
                img = nib.load(cavity)
                assert isinstance(img, nib.Nifti1Image)
                surface = polydata_io.label_nii_to_polydata(img)
            case np.ndarray(), np.ndarray():
                surface = polydata_io.label_to_polydata(cavity, affine)
            case np.ndarray(), None:
                raise ValueError("Affine transformation is required when cavity is a numpy array.")
            case _, _:
                raise ValueError("Invalid input for cavity and affine.")
        logger.debug(f"Loaded cavity surface with {surface.n_points} points and {surface.n_cells} cells.")
        
        c = max(num_components_used, self.n_components)
        k = float((mesh_size(surface) / mesh_size(self.ssm_template)).mean()) * motion_multiplier
        deforms: np.ndarray = k * np.einsum('LPC, LCND -> LPND', self.b[:, :, :c], self.P[:, :c, :, :])
        deforms = einops.rearrange(deforms, 'l p n d -> p (l n) d')
        logger.debug(f"Using {num_components_used} components for deformation, motion_multiplier={k:.3f}, deforms shape: {deforms.shape}")
        
        mesh = deform_surface(
            source_surface=self.ssm_template, 
            target_surface=surface,
        )
        
        deformed_cavities: list[pv.PolyData] = []
        for phase in range(self.n_phases):
            m = mesh.copy()
            # Lable is 1-based, but the index in b and P is 0-based, so need to minus 1.
            m.points += deforms[phase]
            deformed_cavities.append(m)

        landmark = Landmark(mesh, deforms=deforms)
        
        return SSMResult(landmark,  deformed_cavities)

def mesh_size(mesh: pv.PolyData) -> np.ndarray:
    bounds = np.array(mesh.bounds).reshape(3, 2)
    size = np.abs(bounds[:, 1] - bounds[:, 0])
    return size
                
@dataclass
class Landmark:
    mesh: pv.PolyData
    deforms: np.ndarray  # shape = (n_phases, n_labels * n_points, 3), the order of labels is the same as CavityLabel


@dataclass
class SSMResult:
    """
    The result of applying SSM to the template surface.
    
    landmark_surface: The template surface with deformation as point data. 
    point_data["deform_00"], point_data["deform_01"], ... are the deformation of each phase.
    
    deformed_cavities: The deformed cavity surfaces of each phase. 
    The order is the same as the order of deformation in landmark_surface.
    """
    
    landmark: Landmark
    deformed_cavities: list[pv.PolyData]

    def save_vtk(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.landmark.mesh.save(output_dir / "landmark_with_motion.vtk")
        (d := output_dir / "deformed_cavities_dir").mkdir(exist_ok=True)
        for idx, mesh in enumerate(self.deformed_cavities):
            mesh.save(d/f"{idx:02d}.vtk")
    
    def save_nii(self, ref_nii: nib.Nifti1Image, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        to_nii = lambda x: polydata_io.polydata_to_label_nii(x, ref_nii)
            
        nib.save(to_nii(self.landmark.mesh), output_dir / "landmark_with_motion.nii.gz")
        
        (d := output_dir / "deformed_cavities_dir").mkdir(exist_ok=True)
        for idx, mesh in enumerate(self.deformed_cavities):
            nib.save(to_nii(mesh), d/f"{idx:02d}.nii.gz")

        
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
        device: int = 0,
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
    target_surface = target_surface.copy()
    new_cloud = pv.PolyData()
    new_points_all, _ = torchcpd.RigidRegistration(X=target_surface.points, Y=source_surface.points, device=device).register()
    new_points_all, _ = torchcpd.AffineRegistration(X=target_surface.points, Y=new_points_all.cpu().numpy(), device=device).register()
    
    source_surface.points = new_points_all.cpu().numpy()
    for label in sorted(CavityLabel):
        logger.debug(f"process label:{label}")
        source_submesh: pv.PolyData = source_surface.extract_values(label, scalars="label", preference="cell").extract_surface(algorithm=None)    # type: ignore
        target_submesh: pv.PolyData = target_surface.extract_values(label, scalars="label", preference="cell").extract_surface(algorithm=None)    # type: ignore
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