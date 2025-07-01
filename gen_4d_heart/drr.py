from math import radians, floor
from typing import TypeVar
from pathlib import Path
from typing import Literal, Optional
from torch.nn import functional as F

from tqdm import tqdm
import torch
from diffdrr.drr import DRR
from matplotlib import pyplot as plt
from torchio import LabelMap, ScalarImage, Subject
from diffdrr.visualization import plot_drr
from diffdrr.data import canonicalize
import nibabel as nib
from scipy.ndimage import label, center_of_mass
import numpy as np
from monai.networks.blocks.warp import DVF2DDF, Warp
import tifffile

from nibabel.nifti1 import Nifti1Image

from . import NUM_TOTAL_PHASE
from .roi import ROI


def get_delta_degree_at_frame(frame: int, omega: float | int = 75, fps: int = 60) -> float:
    """
    Args:
        frame (int): the frame number, starting from 0
        omega (float | int): degree velocity, default 75 degree per second
        fps (int): frame per second, default 60 fps

    Returns:
        float: the degree of rotation at given frame
    """
    return frame * omega / fps

def get_delta_phase_at_frame(frame: int, fps: int = 60) -> float:
    """
    Args:
        frame (int): the frame number, starting from 0
        fps (int): frame per second, default 60 fps
    
    Returns:
        float: the phase of cardiac cycle at given frame 
    """
    return (frame * NUM_TOTAL_PHASE / fps) % NUM_TOTAL_PHASE

T = TypeVar("T", np.ndarray, torch.Tensor)
def get_dvf_at_phase(dvf_list: list[T], phase: float) -> T:
    """
    Args:
        dvf_list (list[T]): list of dvf at each phase
        phase (float): the phase of cardiac cycle, starting from 0
    
    Returns:
        T: the dvf at given phase
    """
    phase_0 = floor(phase)
    phase_1 = phase_0 + 1
    if phase_1 >= NUM_TOTAL_PHASE:
        phase_1 = 0
    a = phase - phase_0
    b = 1 - a
    dvf_0 = dvf_list[phase_0]
    dvf_1 = dvf_list[phase_1]
    return dvf_0 * b + dvf_1 * a

def get_reorientation(
        orientation_type: Optional[Literal["AP", "PA"]] = "AP"
) -> torch.Tensor:
    # Frame-of-reference change
    if orientation_type == "AP":
        # Rotates the C-arm about the x-axis by 90 degrees
        # Rotates the C-arm about the z-axis by -90 degrees
        reorient = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif orientation_type == "PA":
        # Rotates the C-arm about the x-axis by 90 degrees
        # Rotates the C-arm about the z-axis by 90 degrees
        reorient = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif orientation_type is None:
        # Identity transform
        reorient = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    else:
        raise ValueError(f"Unrecognized orientation {orientation_type}")
    return reorient

def get_drr(
    image: torch.Tensor,
    label: torch.Tensor,
    affine: np.ndarray | None,
    rotations_degree: tuple[float, float, float],
    device: torch.device
) -> torch.Tensor:
    # Ensure tensors are on CPU for ScalarImage/LabelMap creation
    volume = ScalarImage(
        tensor=image.squeeze(0).to(device),
        affine=affine,
    )
    coronary_segmentation = LabelMap(
        tensor=label.squeeze(0).to(device),
        affine=affine,
    )
    
    lung_alpha = 1.0  # -600 <= Hu < 0
    heart_alpha = 0.3  # 0 <= Hu < 500
    bone_alpha = 1.5    # HU >= 600
    coronary_alpha = 10.0

    volume_data = volume.data.to(torch.float32)
    air = torch.where(volume_data <= -600)
    lung = torch.where((-600 < volume_data) & (volume_data <= 0))
    # lung = torch.where((volume_data <= 0))
    heart = torch.where((0 < volume_data) & (volume_data <= 600))
    bone = torch.where(volume_data > 600)
    coronary_segmentation_data = coronary_segmentation.data
    coronary = torch.where(coronary_segmentation_data > 0)

    density = torch.empty_like(volume_data)
    density[air] = volume_data[lung].min()
    density[lung] = volume_data[lung] * lung_alpha
    density[heart] = volume_data[heart] * heart_alpha
    density[bone] = volume_data[bone] * bone_alpha
    density[coronary] = volume_data[coronary].mean() * coronary_alpha
    
    subject = Subject(
        volume = volume,
        mask = coronary_segmentation,
        reorient = get_reorientation("AP"),     # type: ignore
        density = ScalarImage(tensor=density, affine=volume.affine),
        fiducials = None,   # type: ignore
    )
    
    subject = canonicalize(subject)
    
    drr = DRR(
        subject,
        sdd=1800.0,
        height=512,
        delx=0.3,
    ).to(device)
    
    rotations_radians = [radians(rotation) for rotation in rotations_degree]
    rotations = torch.tensor([rotations_radians], device=device)
    translations = torch.tensor([[0.0, 1500.0, 0.0]], device=device)

    img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
    return img

def separate_coronary(coronary: np.ndarray) -> np.ndarray:
    """
    separate coronary to LCA(1) and RCA(2)
    Args:
        coronary (LabelMap): coronary segmentation
    Returns:
        coronary (LabelMap): LCA and RCA
    """
    # Find all connected components
    labeled_array, num_features = label(coronary)  # type: ignore
    
    # If only one component, return original (no separation needed)
    if num_features <= 1:
        return coronary
    
    # Calculate sizes of each component
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
    
    # Find two largest components
    largest_indices = np.argsort(component_sizes)[-2:][::-1] + 1  # +1 to account for skipped 0
    
    region_0 = (labeled_array == largest_indices[0]).astype(np.uint8)
    region_1 = (labeled_array == largest_indices[1]).astype(np.uint8)
    
    center_0 = center_of_mass(region_0)
    center_1 = center_of_mass(region_1)
    
    if center_0[1] > center_1[1]:
        return region_0 * 1 + region_1 * 2
    else:
        return region_1 * 1 + region_0 * 2
    

def generate_rotate_dsa(
    dvf_dir: Path,
    roi_json: Path,
    image_path: Path,
    coronary_path: Path,
    output_dsa_path: Path,
    total_frame: int,
    alpha_start: float = 0.0,
    beta_start: float = 0.,
    # todo add omega, fps
):
    roi = ROI.from_json(roi_json)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dvf_list: list[torch.Tensor] = []
    # zoom_rate = 1 / roi.get_zoom_rate().reshape(1, 3, 1, 1, 1)
    for phase in tqdm(range(NUM_TOTAL_PHASE), desc='loading DVFs'):
        dvf_path = dvf_dir / f'phase_{phase:02d}.nii.gz'
        if not dvf_path.exists():
            raise FileNotFoundError(f"DVF file not found: {dvf_path}")
        dvf = nib.loadsave.load(dvf_path)
        dvf_data = dvf.get_fdata()  # type: ignore
        dvf_tensor = torch.from_numpy(dvf_data)
        dvf_tensor = dvf_tensor.squeeze().permute(3, 0, 1, 2)[None] # (1,3,H,W,D)
        # dvf_tensor = F.interpolate(dvf_tensor, scale_factor=zoom_rate.flatten().tolist(), mode='trilinear', align_corners=False)
        # dvf_tensor = dvf_tensor * torch.from_numpy(zoom_rate)
        dvf_list.append(dvf_tensor)
        
    if len(dvf_list) != NUM_TOTAL_PHASE:
        raise ValueError(f"Expected {NUM_TOTAL_PHASE} DVF files, found {len(dvf_list)}")

    image = nib.loadsave.load(image_path)  
    image = roi.crop_zoom(image, is_label=False)   # type: ignore
    image_tensor = torch.from_numpy(image.get_fdata()).to(device).float()[None, None]
    coronary = nib.loadsave.load(coronary_path)
    coronary = roi.crop_zoom(coronary, is_label=True)  # type: ignore
    # coronary_data = separate_coronary(coronary.get_fdata())       # TODO 当前分辨率太细小,会断裂
    coronary_tensor = torch.from_numpy(coronary.get_fdata()).float().to(device)[None, None]

    
    dvf2ddf = DVF2DDF(num_steps=7, mode="bilinear", padding_mode="zeros").to(device)
    warp_image = Warp(mode="bilinear", padding_mode="zeros").to(device)
    warp_coronary = Warp(mode="nearest", padding_mode="zeros").to(device)

    drr_projections = []
    for frame in tqdm(range(total_frame), desc='generating DRR projections'):
        d_alpha = get_delta_degree_at_frame(frame)
        d_phase = get_delta_phase_at_frame(frame)
        
        with torch.no_grad():
            ddf = dvf2ddf(get_dvf_at_phase(dvf_list, d_phase).to(device)).float()  # Shape (1,3,H,W,D)# (H,W,D,3) -> (3,H,W,D)
            warped_image = warp_image(image_tensor, ddf)
            warped_coronary = warp_coronary(coronary_tensor, ddf)
        
            # Generate DRR projection
            drr_res = get_drr(warped_image, warped_coronary, image.affine, (alpha_start + d_alpha, beta_start, 0), device)
            torch.cuda.empty_cache()
            drr_projections.append(drr_res.squeeze().cpu().numpy())
    
    # Ensure output directory exists and save as multi-frame TIFF
    output_dsa_path.parent.mkdir(parents=True, exist_ok=True)
    drr_np = np.stack(drr_projections)
    drr_np = ((drr_np - drr_np.min()) / (drr_np.max() - drr_np.min()))*255
    tifffile.imwrite(output_dsa_path, 255 - drr_np.astype(np.uint8), imagej=True)


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    def test_generate_rotate_dsa():
        parser = ArgumentParser(description='Generate rotating DSA from 4D heart data')
        parser.add_argument('--dvf_dir', type=Path, required=True,
                          help='Directory containing DVF files (phase_00.nii.gz, etc)')
        parser.add_argument('--roi_json', type=Path, required=True,
                          help='ROI json file containing crop and zoom info')
        parser.add_argument('--image_path', type=Path, required=True,
                          help='Path to base CTA image')
        parser.add_argument('--coronary_path', type=Path, required=True,
                          help='Path to coronary segmentation')
        parser.add_argument('--output_path', type=Path, required=True,
                          help='Output TIFF file path')
        parser.add_argument('--total_frames', type=int, default=120,
                          help='Total frames to generate (default: 120)')
        
        args = parser.parse_args()
        
        print(f"Generating rotating DSA with {args.total_frames} frames...")
        generate_rotate_dsa(
            dvf_dir=args.dvf_dir,
            roi_json=args.roi_json,
            image_path=args.image_path,
            coronary_path=args.coronary_path,
            output_dsa_path=args.output_path,
            total_frame=args.total_frames
        )
        print("Done!")
    
    test_generate_rotate_dsa()
