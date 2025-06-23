from math import radians, floor
from pathlib import Path
from typing import Literal, Optional

import torch
from diffdrr.drr import DRR
from matplotlib import pyplot as plt
from torchio import LabelMap, ScalarImage, Subject
from diffdrr.visualization import plot_drr
from diffdrr.data import canonicalize
import nibabel as nib
from scipy.ndimage import label
import numpy as np
from monai.networks.blocks.warp import DVF2DDF, Warp

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
    return frame * NUM_TOTAL_PHASE % fps

def get_dvf_at_phase(dvf_list: list[np.ndarray], phase: float) -> np.ndarray:
    """
    Args:
        dvf_list (list[torch.Tensor]): list of dvf at each phase
        phase (float): the phase of cardiac cycle, starting from 0
    
    Returns:
        torch.Tensor: the dvf at given phase
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
    affine: torch.Tensor,
    rotations_degree: tuple[float, float, float],
    device: torch.device
) -> torch.Tensor:
    volume = ScalarImage(
        tensor=image,
        affine=affine,
    )
    coronary_segmentation = LabelMap(
        tensor=label,
        affine=affine,
    )
    
    lung_alpha = 1.0  # -800 <= Hu < 0
    heart_alpha = 0.3  # 0 <= Hu < 500
    bone_alpha = 1.0    # HU >= 500
    coronary_alpha = 8.0

    volume_data = volume.data.to(torch.float32)
    air = torch.where(volume_data <= -600)
    lung = torch.where((-600 < volume_data) & (volume_data <= 0))
    # lung = torch.where((volume_data <= 0))
    heart = torch.where((0 < volume_data) & (volume_data <= 500))
    bone = torch.where(volume_data > 500)
    coronary_segmentation_data = coronary_segmentation.data
    coronary = torch.where(coronary_segmentation_data > 0)


    density = torch.empty_like(volume_data)
    density[air] = volume_data[lung].min()
    density[lung] = volume_data[lung] * lung_alpha
    density[heart] = volume_data[heart] * heart_alpha
    density[bone] = volume_data[bone] * bone_alpha
    density[coronary] = volume_data[coronary].mean() * coronary_alpha

    density = -density
    density -= density.min()
    density /= density.max()
    
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
    translations = torch.tensor([[0.0, 1300.0, 0.0]], device=device)

    return drr(rotations, translations, parameterization="euler_angles", convention="ZXY")

def separate_coronary(coronary: LabelMap) -> LabelMap:
    """
    separate coronary to LCA(1) and RCA(2)
    Args:
        coronary (LabelMap): coronary segmentation
    Returns:
        coronary (LabelMap): LCA and RCA
    """
    # Get the underlying tensor data
    coronary_data = coronary.data.numpy()
    
    # Find all connected components
    labeled_array, num_features = label(coronary_data)  # type: ignore
    
    # If only one component, return original (no separation needed)
    if num_features <= 1:
        return coronary
    
    # Calculate sizes of each component
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
    
    # Find two largest components
    largest_indices = np.argsort(component_sizes)[-2:][::-1] + 1  # +1 to account for skipped 0
    
    # Create new segmentation
    new_segmentation = np.zeros_like(coronary_data)
    new_segmentation[labeled_array == largest_indices[0]] = 1  # LCA
    new_segmentation[labeled_array == largest_indices[1]] = 2  # RCA
    
    # Create new LabelMap with the separated segmentation
    new_coronary = LabelMap(
        torch=torch.tensor(new_segmentation, dtype=torch.int64),  # type: ignore
        affine=coronary.affine,
    )
    
    return new_coronary
    

def generate_rotate_dsa(
    dvf_dir: Path,
    roi_json: Path,
    image_path: Path,
    coronary_path: Path,
    output_dsa_path: Path,
    total_frame: int = 180
):
    # Load ROI info
    roi = ROI.from_json(roi_json)
    
    # Load DVF files
    dvf_list = []
    for dvf_path in sorted(dvf_dir.iterdir()):
        if dvf_path.suffix == '.nii.gz':
            dvf = nib.loadsave.load(dvf_path)
            dvf_list.append(dvf.get_fdata())  # type: ignore
    
    # Load base images
    image = ScalarImage(image_path)
    coronary = LabelMap(coronary_path)
    
    # Initialize warping tools
    dvf2ddf = DVF2DDF(num_steps=7, mode="bilinear", padding_mode="zeros")
    warp_image = Warp(mode="bilinear", padding_mode="zeros")
    warp_coronary = Warp(mode="nearest", padding_mode="zeros")
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare output TIFF
    drr_projections = []
    
    for frame in range(total_frame):
        print(f"Processing frame {frame}")
        # Calculate rotation and phase
        d_beta = get_delta_degree_at_frame(frame)
        d_phase = get_delta_phase_at_frame(frame)
        
        # Get DVF at current phase and convert to DDF
        dvf_frame = get_dvf_at_phase(dvf_list, d_phase)
        dvf_tensor = torch.from_numpy(dvf_frame).float().to(device)
        ddf = dvf2ddf(dvf_tensor.unsqueeze(0))
        
        # Restore DDF to original resolution using ROI
        ddf_original = roi.recover(Nifti1Image(ddf.squeeze().cpu().numpy(), affine=np.eye(4)), is_label=False)
        ddf_original_tensor = torch.from_numpy(ddf_original.get_fdata()).float().to(device)
        
        # Warp image and label
        image_tensor = image.data.to(device)
        warped_image = warp_image(image_tensor.unsqueeze(0), ddf_original_tensor.unsqueeze(0))
        
        coronary_tensor = coronary.data.to(device)
        warped_coronary = warp_coronary(coronary_tensor.unsqueeze(0), ddf_original_tensor.unsqueeze(0))
        
        # Generate DRR projection
        rotations = (0, d_beta, 0)  # Rotate around Y-axis
        affine_tensor = torch.from_numpy(image.affine).float().to(device)
        drr = get_drr(warped_image.squeeze(0), 
                     warped_coronary.squeeze(0),
                     affine_tensor,
                     rotations,
                     device)
        
        drr_projections.append(drr.cpu().numpy())
    
    # Save all projections as multi-frame TIFF
    import tifffile
    tifffile.imwrite(output_dsa_path, np.stack(drr_projections), imagej=True)


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
        parser.add_argument('--total_frames', type=int, default=180,
                          help='Total frames to generate (default: 180)')
        
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
