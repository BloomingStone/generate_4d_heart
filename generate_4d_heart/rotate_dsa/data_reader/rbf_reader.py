from typing import Literal, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from pprint import pprint

import numpy as np
import pyvista as pv
from cupyx.scipy.spatial import KDTree
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from monai.networks.blocks.warp import Warp
from einops import rearrange, einsum
from cupyx.scipy import ndimage
import cupy as cp

from generate_4d_heart.roi import ROI
from generate_4d_heart.med_image import MedImage, Original, Centering
from generate_4d_heart.rotate_dsa.data_reader.data_reader import (
    DataReader, DataReaderResult, Coronary, CenteringMode,
    load_nifti, apply_affine, cleanup_gpu_memory
)
from generate_4d_heart.ssm import SSMReader, Landmark
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.types import CoronaryType
from generate_4d_heart import CavityLabel
from generate_4d_heart.rotate_dsa.contrast_simulator import ContrastSimulator
from generate_4d_heart.rotate_dsa.artifacts import ArtifactManager, Artifact


logger = logging.getLogger(__name__)

@dataclass
class RBFReader(DataReader):
    volume_nii: Path
    cavity_nii: Path
    coronary_nii: Path
    
    contrast_simulator: ContrastSimulator
    
    ssm_reader: SSMReader= field(default_factory=SSMReader)
    
    device: torch.device = field(default_factory= lambda: torch.device("cuda:0"))
    num_components_used: int = 1
    motion_multiplier: float = 1.3
    attenuation_transition: float = 5.0
    
    artifact_manager: ArtifactManager | None = None
    
    centering_mode: CenteringMode = "coronary_center"
    
    centerline_constraint_enabled: bool = True
    """
    If True (default), the RBF control-point displacements are projected
    to first-order preserve coronary centreline edge lengths, preventing
    stretching/shrinking of guide wires, stents, and markers.
    Set to False to disable the constraint and use the raw SSM motion.
    """
    
    background_nii: Path | None = None
    """
    Optional whole-chest background CT (same world coordinates as volume_nii).
    If provided, used as the first static layer for DRR.
    The region occupied by volume_nii is zeroed out to avoid double-counting.
    """
    
    feather_width: float = 10
    """
    Feathering width (in background voxels) for the inward feathering of the volume_nii
    """

    def __post_init__(self):
        print(f"\nInitializing RBFReader with \n\t- volume_nii={self.volume_nii} \n\t- cavity_nii={self.cavity_nii} \n\t- coronary_nii={self.coronary_nii}")
        print(f"\t- Centering mode: {self.centering_mode}")
        print(f"\t- Using contrast_simulator: {self.contrast_simulator}")
        if self.background_nii is not None:
            print(f"\t- Using background_nii: {self.background_nii}")
        if self.artifact_manager is not None:
            print(f"\t- Using artifact_manager:\n\t", end="")
            pprint(self.artifact_manager.config)
        
        logger.info(f"Loading data from {self.volume_nii}, {self.cavity_nii}, {self.coronary_nii}")
        self.n_phases = self.ssm_reader.n_phases
        
        print("Loading NIfTI files...")
        volume, affine = load_nifti(self.volume_nii)
        assert affine is not None
        cavity, cavity_affine = load_nifti(self.cavity_nii, is_label=True)
        cavity_np = cavity.squeeze().cpu().numpy()
        coronary, coronary_affine = load_nifti(self.coronary_nii, is_label=True)
        assert np.allclose(affine, cavity_affine)
        assert np.allclose(affine, coronary_affine)        
        self._origin_volume_size = volume.shape[2:]   #type: ignore
        self._origin_volume_affine = affine
        
        print("Initializing origin data...")
        self.origin_data = self._Data.init_from_volume(
            volume, cavity, coronary, affine, self
        )
        
        # ── Load and preprocess optional background, apply inward feathering to volume ──
        # NOTE: must be after init_from_volume so origin_data's volumes are already in mu space
        # (HU→mu).  Both sides feather in the same mu domain — no cross-domain trick needed.
        self._load_background(self.origin_data, affine)
        
        print("Initializing ROI from cavity and cropping data...")
        self.roi = ROI.get_from_cavity_np(cavity_np, affine, padding=20)
        self.cropped_data = self.origin_data.crop_by_roi(self.roi)
        
        # Only cropped_data needs to be on GPU for per-frame warp + render.
        # origin_data is kept on CPU to avoid wasting ~1 GB+ GPU memory:
        #   - get_static_data() clones volume data (CPU→GPU on demand in RotateDSA.run())
        #   - get_dynamic_data() accesses mesh_original (CPU pyvista)
        self.cropped_data = self.cropped_data.to_device(self.device)
        
        print("Generating pericardium mask and attenuation mask...")
        _, pericardium_mask = generate_pericardium(cavity_np, coronary.squeeze().cpu().numpy())
        pericardium_mask = self.roi.crop_on_data(pericardium_mask)
        with cp.cuda.Device(self.device.index):
            external_dist = ndimage.distance_transform_edt(1 - cp.asarray(pericardium_mask))
            attenuation_mask = cp.asnumpy(cp.exp(-external_dist / self.attenuation_transition))  # 从心包外部开始，距离越远运动衰减越强，距离为transition时衰减为1/e #type: ignore
        self.attenuation_mask = torch.from_numpy(attenuation_mask).to(self.device, torch.float32)[None, None]  # (1, 1, *shape_after_crop)
        
        print("Loading SSM result and initializing RBF...")
        ssm_result = self.ssm_reader.load(
            cavity=cavity_np,
            affine=cavity_affine,
            num_components_used=self.num_components_used,
            motion_multiplier=self.motion_multiplier
        )

        # Extract coronary centre-lines for length-preserving RBF constraints
        print("Extracting coronary centre-lines for RBF length constraint...")
        centerline_lca = self.origin_data.lca.get_coronary_central_line("world")
        centerline_rca = self.origin_data.rca.get_coronary_central_line("world")
        self.rbf = KDTreeRBF(
            ssm_result.landmark,
            centerlines=[centerline_lca, centerline_rca],
            centerline_downsample=3,
            centerline_constraint_enabled=self.centerline_constraint_enabled,
            device=self.device,
        ).to(self.device).eval()
        
        print("Caching grid points for RBF-based warping...")
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
        
        # Pre-compute KDTree neighbour indices for the static dense grid.
        # This avoids GPU→CPU transfer of 2.65M points + CPU KDTree query per frame.
        print("Pre-computing KDTree indices for dense grid...")
        self.rbf.cache_points(coords_zoomed)  # stores original (X, Y, Z, 3) shape
        self.rbf.cache_kdtree_indices(coords_zoomed.reshape(-1, 3))
        
        self.affine_crop = torch.from_numpy(self.roi.affine_after_crop).to(self.device, torch.float32)
        self.affine_crop_inv: torch.Tensor = torch.linalg.inv(self.affine_crop)
        
        # ignore warning form monai about grid_sample, using defult pytorch grid_sample rather than monai's 
        # cuda implementation, which not included in conda source
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="monai.networks.blocks.warp")
        self.label_warper = Warp(mode="nearest", padding_mode="border").to(self.device).eval()
        self.image_warper = Warp(mode="bilinear", padding_mode="border").to(self.device).eval()
        
        print("RBFReader initialized successfully.")

    def cleanup(self) -> None:
        """Release GPU resources held by this reader. Safe to call multiple times."""
        # Move GPU tensors to CPU so they are freed from GPU memory
        if hasattr(self, 'cropped_data') and self.cropped_data is not None:
            try:
                self.cropped_data = self.cropped_data.to_device(torch.device('cpu'))
            except Exception:
                pass
        if hasattr(self, 'coords_zoomed') and self.coords_zoomed is not None:
            self.coords_zoomed = self.coords_zoomed.cpu()
        if hasattr(self, 'attenuation_mask') and self.attenuation_mask is not None:
            self.attenuation_mask = self.attenuation_mask.cpu()
        if hasattr(self, 'affine_crop') and self.affine_crop is not None:
            self.affine_crop = self.affine_crop.cpu()
        if hasattr(self, 'affine_crop_inv') and self.affine_crop_inv is not None:
            self.affine_crop_inv = self.affine_crop_inv.cpu()
        cleanup_gpu_memory()

    def __del__(self) -> None:
        """Defensive cleanup when the reader is garbage collected."""
        self.cleanup()

    @torch.no_grad()
    def _load_background(
        self,
        origin_data: DataReader._Data,
        affine: np.ndarray,
    ) -> None:
        """
        Load and process the optional background CT, and apply inward feathering to the volume.

        This method:
          1. Loads the background NIfTI and applies contrast preprocessing.
          2. Computes the volume's AABB in background voxel coordinates.
          3. Applies **inward feathering** to *volume* (attenuates near the AABB boundary)
          4. Zeroes the AABB interior in the background and applies **outward feathering**
             (cosine ramp) so the background smoothly fades out at the boundary.
        """
        self._background_img: MedImage[Original] | None = None
        if self.background_nii is None:
            return
        
        mempool = cp.get_default_memory_pool()
        pinned_mem_pool = cp.get_default_pinned_memory_pool()

        print(f"Loading background from {self.background_nii}...")
        bg_vol, bg_affine = load_nifti(self.background_nii)  # (1,1,D,H,W)
        dummy_cavity = torch.zeros_like(bg_vol, dtype=torch.uint8)

        bg_vol = self.contrast_simulator.preprocess(bg_vol, dummy_cavity, bg_affine)

        # Compute volume's AABB in background voxel coords
        W, H, D = self._origin_volume_size
        corners_voxel = np.array([
            [0, 0, 0, 1], [W-1, 0, 0, 1], [0, H-1, 0, 1], [0, 0, D-1, 1],
            [W-1, H-1, 0, 1], [W-1, 0, D-1, 1], [0, H-1, D-1, 1], [W-1, H-1, D-1, 1],
        ], dtype=np.float32).T
        corners_world = self._origin_volume_affine @ corners_voxel
        inv_bg = np.linalg.inv(bg_affine)
        corners_bg = inv_bg @ corners_world
        bg_vox_min = corners_bg[:3].min(axis=1).astype(int)
        bg_vox_max = corners_bg[:3].max(axis=1).astype(int) + 1
        bg_shape = np.array(bg_vol.shape[-3:])
        bg_vox_min = np.maximum(bg_vox_min, 0)
        bg_vox_max = np.minimum(bg_vox_max, bg_shape)

        feather = self.feather_width  # voxels in background space

        # Initialize CuPy/torch variables before try so finally can safely del them
        vol_grid = vol_flat = vol_h = world_pts = bg_pts = None
        bg_data = bg_original = aabb_mask = outward_mask = None
        feather_mask = dist = dist_cp = w = w_cos = None

        try:
            # ── Inward feathering on the volume ──
            sample_vol = next(iter(origin_data.coronary.values())).volume.data
            D_vol, H_vol, W_vol = sample_vol.shape[-3:]

            # Step 1–5: CuPy-accelerated grid → bg-voxel distance computation
            i = cp.arange(D_vol)
            j = cp.arange(H_vol)
            k = cp.arange(W_vol)
            vol_grid = cp.stack(cp.meshgrid(i, j, k, indexing='ij'), axis=-1)  # (D, H, W, 3)

            # Step 2: flatten to (N, 3) and add homogeneous coordinate
            vol_flat = vol_grid.reshape(-1, 3).astype(cp.float32)
            vol_h = cp.pad(vol_flat, ((0, 0), (0, 1)), constant_values=1)  # (N, 4)

            # Step 3: vol_voxel → world → bg_voxel
            affine_cp = cp.asarray(affine)
            inv_bg_cp = cp.asarray(inv_bg)
            world_pts = vol_h @ affine_cp.T
            bg_pts = world_pts @ inv_bg_cp.T
            bg_pts = bg_pts[:, :3]

            # Step 4: distance from each point to the nearest AABB face
            bg_vox_min_cp = cp.asarray(bg_vox_min)
            bg_vox_max_cp = cp.asarray(bg_vox_max)
            d_min = bg_pts - bg_vox_min_cp[None, :]
            d_max = bg_vox_max_cp[None, :] - bg_pts
            d_face = cp.minimum(d_min, d_max)
            d_boundary = d_face.min(axis=-1)
            d_boundary = cp.maximum(d_boundary, 0)

            # Step 5: cosine inward ramp — 0 at the boundary, 1 at `feather` or deeper
            w = cp.clip(d_boundary / feather, 0, 1).astype(cp.float32)
            atten_cos = 0.5 * (1.0 - cp.cos(cp.pi * w))
            atten = cp.asnumpy(atten_cos.reshape(D_vol, H_vol, W_vol))

            # Step 6: apply inward attenuation directly in mu space, in-place.
            for cor in origin_data.coronary.values():
                cor.volume.data[0, 0] *= torch.from_numpy(atten)

            # ── Process background: zero AABB interior + outward feathering ──
            #
            # Complement to the inward volume feathering above.
            #
            # Key: the background zeroing region is the volume's AABB CONTRACTED by
            # `feather` voxels on each side.  This makes the two cosine ramps
            # complementary → vol + bg = 1 everywhere in the transition zone:
            #
            #   contracted AABB                    original AABB
            #   ──────────●────────────────────────●─────────────
            #      inner ←│     ← feather →        │→ outer 
            #   vol:  1───┘                        └─0  (inward cosine)
            #   bg:   0───┐                        ┌─1  (outward cosine)
            #   total:    1                        1
            bg_zero_min = np.maximum(bg_vox_min + feather, 0).astype(int)
            bg_zero_max = np.minimum(bg_vox_max - feather, bg_shape).astype(int)
            for axis in range(3):
                if bg_zero_max[axis] <= bg_zero_min[axis]:
                    bg_zero_min[axis] = bg_vox_min[axis]
                    bg_zero_max[axis] = bg_vox_min[axis] + 1

            bg_data = bg_vol.squeeze()
            bg_original = bg_data.clone()

            bg_data[bg_zero_min[0]:bg_zero_max[0],
                    bg_zero_min[1]:bg_zero_max[1],
                    bg_zero_min[2]:bg_zero_max[2]] = 0.0

            aabb_mask = torch.zeros(bg_shape.tolist(), dtype=torch.bool)
            aabb_mask[bg_zero_min[0]:bg_zero_max[0],
                    bg_zero_min[1]:bg_zero_max[1],
                    bg_zero_min[2]:bg_zero_max[2]] = True
            outward_mask = ~aabb_mask
            # CuPy-accelerated distance transform on GPU
            with cp.cuda.Device(self.device.index):
                dist_cp = ndimage.distance_transform_edt(cp.asarray(outward_mask.numpy()))
                dist = torch.from_numpy(cp.asnumpy(dist_cp))
            feather_mask = (dist > 0) & (dist <= feather)
            w = dist[feather_mask] / feather
            w_cos = (0.5 * (1 - torch.cos(np.pi * w))).to(bg_data.dtype)
            bg_data[feather_mask] = bg_original[feather_mask] * w_cos

            bg_volume = bg_data[None, None].to(torch.float32).cpu()

            self._background_img = MedImage(
                data=bg_volume,
                affine=bg_affine.copy(),
                coord=Original(),
            )

            print(f"Background: shape={bg_vol.shape[-3:]}, "
                    f"volume AABB bg-voxel: {bg_vox_min}→{bg_vox_max}")
        finally:
            # Clean up CuPy memory to avoid OOM in subsequent tests
            del vol_grid, vol_flat, vol_h, world_pts, bg_pts
            del bg_data, bg_original, aabb_mask, outward_mask
            del feather_mask, dist, dist_cp, w, w_cos
            mempool.free_all_blocks()
            pinned_mem_pool.free_all_blocks()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _warp_for_phase(
        self, 
        cor_type: CoronaryType, 
        phase: CardiacPhase, 
        global_time: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Return warped (volume, cavity_label, coronary_label) recovered to original space for given phase
        形变都在世界坐标下得到，然后再还原回 ROI 裁剪缩放后的体素坐标得到形变场，并同样在此坐标下进行体积和标签的 warping（为了减少
        计算量），最后再还原回原始空间。
        """
        dx: torch.Tensor = self.rbf(phase)  # uses pre-cached dense grid + KDTree indices
        dx = rearrange(dx, "x y z c -> 1 c x y z")
        dx = F.interpolate(
            dx, 
            size=self.roi.shape_after_crop, 
            mode="trilinear", 
            align_corners=False
        )  # (1, 3, *shape_after_crop)  in world coordinate

        R_world_to_voxel = self.affine_crop_inv[:3, :3]        
        dx = einsum(dx, R_world_to_voxel, "b c x y z, c i -> b i x y z")  # in voxel coordinate

        # Attenuate the movement away from the pericardium to avoid unrealistic deformation in the far field
        dx = dx * self.attenuation_mask  

        # forward deformation field -> inverse deformation field, still in voxel coordinate
        dx = invert_displacement_field(dx)

        # extract label and volume for the given coronary type, and apply contrast simulation if needed
        cropped_cavity = self.cropped_data.cavity.data
        cropped_cor_label = self.cropped_data.coronary[cor_type].label.data

        cropped_volume = self.cropped_data.coronary[cor_type].volume.data

        if self.contrast_simulator.contrast_change_over_time:
            cropped_volume = self.contrast_simulator.simulate_with_time(
                global_time,
                cropped_volume,
                cropped_cavity,
                cropped_cor_label,
                self.cropped_data.coronary[cor_type].centering_affine
            )

        # apply warping and recover to original space, for both volume and labels
        def _warp(tensor: Tensor, mode: Literal["volume", "cavity", "coronary"]) -> Tensor:
            if mode == "volume":
                warper = self.image_warper
                final_dtype = torch.float32
            else:
                warper = self.label_warper
                if mode == "cavity":
                    final_dtype = torch.uint8
                elif mode == "coronary":
                    final_dtype = torch.bool
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
            return warper(tensor.to(self.device, torch.float32), dx).to(final_dtype)

        cropped_cavity = _warp(self.cropped_data.cavity.data, mode="cavity")
        volume = _warp(cropped_volume, mode="volume")
        cropped_cor_label = _warp(cropped_cor_label, mode="coronary")
        
        return volume, cropped_cavity, cropped_cor_label

    @torch.no_grad()
    def get_static_data(
        self,
        coronary_type: CoronaryType | Literal["LCA", "RCA"],
    ) -> list[MedImage[Centering]]:
        """
        输出多层静态volume(背景、心腔等)。每层都将下一层占据的区域置为零（或在边缘增加羽化，以处理分辨率不同导致的边缘无法对其问题）
        因为 X射线投影/DRR 是线性积分， 所以可以将每层的投影结果相加，得到最终的投影结果。
        返回结果应当在 Centering 坐标下。
        """
        cor_type = CoronaryType(coronary_type)
        cor = self.origin_data.coronary[cor_type]  # before cropping, match static volume
        
        layers: list[MedImage[Centering]] = []

        # ── Layer 1: Background (if available) ──
        if self._background_img is not None:
            layers.append(
                self._background_img.to_coord(cor.centering_coord)
            )

        # ── Layer 2: Static heart (dynamic ROI zeroed) ──
        (x0, x1), (y0, y1), (z0, z1) = self.roi.crop_box
        heart_vol = cor.volume.data.clone()
        heart_vol[..., x0:x1, y0:y1, z0:z1] = 0.0
        layers.append(
            MedImage(
                data=heart_vol,
                affine=cor.centering_affine,
                coord=cor.centering_coord,
            ))

        return layers

    @torch.no_grad()
    def get_dynamic_data(
        self,
        phase: CardiacPhase,
        coronary_type: CoronaryType | Literal["LCA", "RCA"],
        global_time: float = 0.0,
    ) -> DataReaderResult:
        cor_type = CoronaryType(coronary_type)

        # 冠脉mesh的同样形变先在原始世界坐标下进行，然后再平移到 centering affine 定义的坐标下，这样可以保证冠脉mesh和体积数据最终在投影坐标下能够正确对齐
        cor_mesh = self.origin_data[cor_type].mesh_original.copy()
        mesh_points = torch.from_numpy(cor_mesh.points).to(self.device, torch.float32)
        mesh_points += self.rbf(phase, mesh_points)     # moving in world coordinate
        cor_mesh.points = mesh_points.cpu().numpy()

        # Deform artifact meshes (same RBF displacement, world coordinates)
        artifacts_deformed: list[Artifact] | None = None
        if self.origin_data[cor_type].artifacts_original is not None:
            artifacts_deformed = []
            for art in self.origin_data[cor_type].artifacts_original:  # type: ignore[union-attr]
                art_mesh = art.mesh.copy()
                art_pts = torch.from_numpy(np.array(art_mesh.points)).to(self.device, torch.float32)
                art_pts += self.rbf(phase, art_pts)   # moving in world coordinate
                art_mesh.points = art_pts.cpu().numpy()
                artifacts_deformed.append(Artifact(mesh=art_mesh, mu=art.mu, name=art.name))

        volume_data, cavity_data, cor_label_data = self._warp_for_phase(cor_type, phase, global_time)

        _cropped_data = self.cropped_data[cor_type]
        return DataReaderResult(
            phase=phase,
            cavity=MedImage(data=cavity_data, affine=_cropped_data.original_affine, coord=Original()),
            coronary=Coronary(
                type=cor_type,
                volume=MedImage(data=volume_data, affine=_cropped_data.original_affine, coord=Original()),
                label=MedImage(data=cor_label_data, affine=_cropped_data.original_affine, coord=Original()),
                centering_coord=_cropped_data.centering_coord,
                mesh_original=cor_mesh,
                artifacts_original=artifacts_deformed,
            )
        )
    
    @torch.no_grad()
    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        return self.get_combined_data(CardiacPhase(0), coronary_type)

    @property
    def lca_centering_affine(self) -> np.ndarray:
        return self.origin_data.lca.centering_affine
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        return self.origin_data.rca.centering_affine
    
    @property
    def volume_size(self) -> tuple[int, int, int]:
        return self.roi.original_shape
    
    @property
    def volume_affine(self) -> np.ndarray:
        return self.roi.original_affine

    @property
    def meta(self) -> dict[str, Any]:
        source_files = {
            "volume_nii": str(self.volume_nii),
            "cavity_nii": str(self.cavity_nii),
            "coronary_nii": str(self.coronary_nii),
            "background_nii": str(self.background_nii) if self.background_nii is not None else None,
        }
        if self.background_nii is not None:
            source_files["background_nii"] = str(self.background_nii)
        return {
            "n_phases": self.n_phases,
            "centering_mode": self.centering_mode,
            "volume_size": self.volume_size,
            "volume_affine": self.volume_affine.tolist(),
            "source_files": source_files,
            "num_components_used": self.num_components_used,
            "motion_multiplier": self.motion_multiplier,
            "attenuation_transition": self.attenuation_transition,
            "centerline_constraint_enabled": self.centerline_constraint_enabled,
            "feather_width": self.feather_width,
            "contrast_simulator": self.contrast_simulator.meta if self.contrast_simulator is not None else None,
            "artifact_manager": self.artifact_manager.config.meta if self.artifact_manager is not None else None,
        }

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
    offset_points = points + padding_voxel * np.array(normals)

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
        n_ctrl_pts: int = 500,
        centerlines: list[np.ndarray] | None = None,
        centerline_downsample: int = 3,
        centerline_constraint_enabled: bool = True,
        device: torch.device | None = None,
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
        
        # Build CuPy KDTree on the target GPU device
        if device is not None and device.type == 'cuda':
            with cp.cuda.Device(device.index):
                self.tree = KDTree(cp.asarray(ctrl_pts))
        else:
            self.tree = KDTree(cp.asarray(ctrl_pts))
        
        self._copy_stream: torch.cuda.Stream|None = None
        self.cached_pts: torch.Tensor|None  = None
        self._cached_pts_shape: tuple[int, ...] | None = None
        """Original shape before flattening cached_pts, used to reshape the forward() output."""
        self._cached_idx: torch.Tensor | None = None
        """Pre-computed KDTree neighbour indices for cached_pts to avoid per-frame CPU query + GPU→CPU transfer."""
        
        # ── Optimize ctrl_disps with centerline edge-length constraints ──
        if centerlines is not None and centerline_constraint_enabled:
            self._constrain_centerline_lengths(centerlines, centerline_downsample)
    
    def cache_points(self, points: np.ndarray):
        device = self.ctrl_pts.device
        self._cached_pts_shape = points.shape  # original shape, e.g. (X, Y, Z, 3)
        self.cached_pts = torch.from_numpy(points).to(device, torch.half, non_blocking=True)
    
    def cache_kdtree_indices(self, points_np: np.ndarray):
        """Pre-compute and store KDTree neighbour indices using CuPy GPU KDTree + DLPack. Skips the CPU query in forward()."""
        N = points_np.shape[0]
        idx_list = []
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            _, idx = self.tree.query(
                cp.asarray(points_np[start:end]), k=self.k,
                distance_upper_bound=self.radius,
            )
            if self.k == 1:
                idx = idx[:, None]  # type: ignore
            idx_list.append(torch.tensor(idx))
        self._cached_idx = torch.cat(idx_list, dim=0)

    @torch.no_grad()
    def _constrain_centerline_lengths(
        self,
        centerlines: list[np.ndarray],
        downsample_step: int = 3,
    ):
        """
        Optimize ``ctrl_disps`` for all phases to *first-order* preserve
        centreline edge lengths.

        For each edge :math:`e_i = a_{i+1} - a_i` we enforce

        .. math::
            e_i \\cdot (v_{i+1} - v_i) = 0

        where :math:`v_i = \\sum_j w_{ij} D_j` is the RBF displacement.

        The original displacement :math:`c_{\\text{orig}}` (stacked per-phase)
        is projected onto the null-space of the constraint matrix :math:`A`
        via the closed-form projection

        .. math::
            c' = c - A^T (A A^T + \\lambda I)^{-1} A c .
        """
        device = self.ctrl_pts.device
        ctrl_pts_f = self.ctrl_pts.float()  # (M, 3)
        M = ctrl_pts_f.shape[0]
        sigma2 = 2 * self.sigma ** 2

        # ── 1. Downsample & upload centreline segments ──
        cl_tensors: list[torch.Tensor] = []
        for cl in centerlines:
            sub = cl[::downsample_step]
            if len(sub) >= 2:
                cl_tensors.append(torch.from_numpy(sub).to(device, torch.float32))

        if not cl_tensors:
            return

        # ── 2. Build constraint matrix A ∈ R^{K × 3M} ──
        A_rows: list[torch.Tensor] = []
        total_edges = 0

        for cl_pts in cl_tensors:  # (N_s, 3)
            N = cl_pts.shape[0]
            edges = cl_pts[1:] - cl_pts[:-1]  # (N-1, 3)

            # Pairwise distances  →  Gaussian weights (same scheme as forward())
            dist = torch.cdist(cl_pts, ctrl_pts_f)                     # (N, M)
            mask = (dist <= self.radius)
            w = torch.exp(-dist.square() / sigma2) * mask
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-5)
            w = w / w_sum                                               # (N, M)

            # Δw for each edge
            dw = w[1:] - w[:-1]                                         # (N-1, M)

            # A_seg[k, 3j+c] = dw[k,j] · edges[k,c]
            A_seg = (dw.unsqueeze(1) * edges.unsqueeze(2)).reshape(N - 1, 3 * M)
            A_rows.append(A_seg)
            total_edges += N - 1

        A = torch.cat(A_rows, dim=0)  # (K, 3M)  float32
        K = A.shape[0]

        # ── 3. Precompute A A^T (float64 for stability) ──
        A_f64 = A.to(torch.float64)                                  # (K, 3M)
        ATA = A_f64 @ A_f64.T                                        # (K, K)
        # Tiny regularisation only to handle exact rank deficiency
        reg = torch.eye(K, device=device, dtype=torch.float64) * 1e-10
        ATA_reg = ATA + reg

        print(f"KDTreeRBF: constraining {K} centreline edge-lengths "
              f"across {M} control points for {self.n_phases} phases")

        # ── 4. Compute residual BEFORE projection (float32, for reference) ──
        ctrl_disps_f32 = self.ctrl_disps.float().clone()            # (P, M, 3)
        residual_before = 0.0
        for phase in range(self.n_phases):
            c_test = ctrl_disps_f32[phase].reshape(-1)               # (3M,)
            residual_before += (A @ c_test).norm().item()
        residual_before /= self.n_phases

        # ── 5. Project each phase (solve in float64), then rescale to preserve
        #    original displacement magnitude.
        #    Rescaling is safe here: A(α c') = α A c' = 0, so the edge-length
        #    constraint remains exactly satisfied after uniform scaling.
        norm_ratios: list[float] = []
        for phase in range(self.n_phases):
            c_orig_f64 = ctrl_disps_f32[phase].reshape(-1).to(torch.float64)  # (3M,)
            orig_norm = c_orig_f64.norm().item()
            
            rhs = A_f64 @ c_orig_f64                                        # (K,)
            y = torch.linalg.solve(ATA_reg, rhs)                            # (K,)
            c_new_f64 = c_orig_f64 - A_f64.T @ y                            # (3M,)
            
            # Rescale to restore original displacement magnitude
            new_norm = c_new_f64.norm().item()
            if new_norm > 1e-12:
                c_new_f64 = c_new_f64 * (orig_norm / new_norm)
            
            norm_ratios.append(new_norm / orig_norm if orig_norm > 0 else 1.0)
            
            ctrl_disps_f32[phase] = c_new_f64.float().reshape(M, 3)

        # ── 6. Compute residual AFTER projection (float32, before half-cast) ──
        residual_after = 0.0
        for phase in range(self.n_phases):
            c_test = ctrl_disps_f32[phase].reshape(-1)
            residual_after += (A @ c_test).norm().item()
        residual_after /= self.n_phases

        # ── 7. Log displacement magnitude change ──
        mean_ratio = float(np.mean(norm_ratios))
        min_ratio = float(np.min(norm_ratios))
        max_ratio = float(np.max(norm_ratios))

        # ── 8. Write back ──
        self.ctrl_disps.copy_(ctrl_disps_f32.half())

        print(f"KDTreeRBF: mean constraint residual: "
              f"{residual_before:.4f} → {residual_after:.6f} (float32), "
              f"displacement norm ratio (pre-rescale): {mean_ratio:.4f} "
              f"[{min_ratio:.4f}, {max_ratio:.4f}] → rescaled to 1.0, "
              f"half-precision stored")

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
        idx0 = phase.lower_index(self.n_phases)
        idx1 = phase.upper_index(self.n_phases)
        w = float(phase) * self.n_phases - idx0
        disp = w * self.ctrl_disps[idx1] + (1 - w) * self.ctrl_disps[idx0]
        
        device = self.ctrl_pts.device
        
        match points:
            case None:
                assert self.cached_pts is not None, "points is None and cached_pts is None, please call cache_points first"
                points_gpu = self.cached_pts
                points_np = None  # use cached KDTree indices
            case torch.Tensor():
                points_np = points.cpu().numpy()
                points_gpu = points.to(device, torch.half, non_blocking=True)
            case np.ndarray():
                points_gpu = torch.from_numpy(points).to(device, torch.half, non_blocking=True)
                points_np = points.copy()
            case _:
                raise TypeError(f"Unsupported type for points: {type(points)}")
        
        shape = points_np.shape if points_np is not None else points_gpu.shape
        assert shape[-1] == 3, f"Last dimension of points must be 3, but got {shape[-1]}"
        
        if points is None and self._cached_idx is not None:
            # Fast path: use pre-computed KDTree indices, skip CPU query + GPU→CPU transfer
            points_gpu = points_gpu.reshape(-1, 3)
            N = points_gpu.shape[0]
            motion = torch.zeros((N, 3), dtype=torch.half, device=device)
            prev: tuple[int, int, torch.Tensor] | None = None
            for start in range(0, N, self.chunk_size):
                end = min(start + self.chunk_size, N)
                idx_gpu = self._cached_idx[start:end].to(device, non_blocking=True)
                self._update_motion_by_prev(points_gpu, disp, motion, prev)
                prev = (start, end, idx_gpu)
            self._update_motion_by_prev(points_gpu, disp, motion, prev)
        else:
            # Slow path: run KDTree query on GPU via CuPy + DLPack
            points_np_flat = points_np.reshape(-1, 3)  # type: ignore
            points_gpu = points_gpu.reshape(-1, 3)
            N = points_gpu.shape[0]
            motion = torch.zeros((N, 3), dtype=torch.half, device=device)
            prev = None
            for start in range(0, N, self.chunk_size):
                end = min(start + self.chunk_size, N)
                _, idx = self.tree.query(
                    cp.asarray(points_np_flat[start:end]), k=self.k,
                    distance_upper_bound=self.radius,
                )
                if self.k == 1:
                    idx = idx[:, None]  # type: ignore
                idx_gpu = torch.tensor(idx)
                self._update_motion_by_prev(points_gpu, disp, motion, prev)
                prev = (start, end, idx_gpu)
            self._update_motion_by_prev(points_gpu, disp, motion, prev)
        
        result = motion.float()
        if points is None and self._cached_pts_shape is not None:
            result = result.reshape(self._cached_pts_shape)
        else:
            result = result.reshape(shape)
        return result

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
