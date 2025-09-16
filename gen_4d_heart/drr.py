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
from diffdrr.data import canonicalize
import nibabel as nib
import numpy as np
from monai.networks.blocks.warp import DVF2DDF, Warp
from monai.transforms import dilate
import tifffile
import cv2
import cupy as cp
from cupyx.scipy.ndimage import label, center_of_mass, distance_transform_edt
from torch.utils.dlpack import from_dlpack, to_dlpack

from . import NUM_TOTAL_PHASE, LV_LABEL
from .roi import ROI
import numpy as np
import imageio.v2 as imageio


def get_delta_degree_at_frame(frame: int, omega: float | int = 75, fps: int | float = 60) -> float:
    """
    Args:
        frame (int): the frame number, starting from 0
        omega (float | int): degree velocity, default 75 degree per second
        fps (int): frame per second, default 60 fps

    Returns:
        float: the degree of rotation at given frame
    """
    return frame * omega / fps

def get_delta_phase_at_frame(frame: int, fps: int | float = 60) -> float:
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
    mean_hu_at_coronary: float,
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
    
    lung_alpha = 0.7  # -600 <= Hu < 0
    heart_alpha = 0.5  # 0 <= Hu < 500
    bone_alpha = 1.5    # HU >= 600
    coronary_alpha = 13.0

    volume_data = volume.data.to(torch.float32)
    air = torch.where(volume_data <= -600)
    lung = torch.where((-600 < volume_data) & (volume_data <= 0))
    heart = torch.where((0 < volume_data) & (volume_data <= 600))
    bone = torch.where(volume_data > 600)
    coronary_segmentation_data = coronary_segmentation.data
    coronary = torch.where(coronary_segmentation_data > 0)

    density = torch.empty_like(volume_data)
    density[air] = volume_data[lung].min()
    density[lung] = volume_data[lung] * lung_alpha
    density[heart] = volume_data[heart] * heart_alpha
    density[bone] = volume_data[bone] * bone_alpha
    density[coronary] = mean_hu_at_coronary * coronary_alpha
    
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

def save_frames(output_tif_path: Path, frames_np: np.ndarray, fps) -> None:
    output_tif_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_tif_path, frames_np, imagej=True)
    _, w, h = frames_np.shape
    
    # Save as mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_tif_path.with_suffix(".mp4")), fourcc, fps, (w, h), isColor=False)
    for frame in frames_np:
        video_writer.write(frame)
    video_writer.release()
    

@torch.no_grad()
def generate_rotate_dsa(
    dvf_dir: Path,
    roi_json: Path,
    image_path: Path,
    coronary_path: Path,
    cavity_path: Path,
    output_dsa_path: Path,
    total_frame: int,
    alpha_start: float = 30.0,
    beta_start: float = 0.,
    fps: float = 60.0,
    # todo add omega, fps
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dvf2ddf = DVF2DDF(num_steps=7, mode="bilinear", padding_mode="zeros").to(device)
    warp_image = Warp(mode="bilinear", padding_mode="zeros").to(device)
    warp_coronary = Warp(mode="nearest", padding_mode="zeros").to(device)
    roi = ROI.from_json(roi_json)
    
    def load_tensor(_path: Path) -> torch.Tensor:
        img = roi.crop(nib.loadsave.load(_path))  # type: ignore
        tensor = torch.from_numpy(img.get_fdata()).to(device).float()
        return tensor, img.affine

    image, image_affine = load_tensor(image_path)
    coronary, _ = load_tensor(coronary_path)
    cavity, _ = load_tensor(cavity_path)
    mean_Hu_at_coronary = image[coronary==1].mean().item()
    
    dilate_coronary = dilate(coronary[None, None], filter_size=9).squeeze()
    # use dvf at the LV part to enhance the movement of coronary
    lv_cp = cp.from_dlpack(to_dlpack(cavity==LV_LABEL)).astype(cp.bool_)  # (W,H,D), bool
    coronary_cp = cp.from_dlpack(to_dlpack(dilate_coronary)).astype(cp.bool_)  # (W,H,D), bool
    dist, (ix, iy, iz) = distance_transform_edt(~lv_cp, return_distances=True, return_indices=True)  # for now ix.shape = (W,H,D)
    jx, jy, jz = cp.where(coronary_cp)      # (jx, jy, jz) is the coordinate of coronay, jx.shape = (N,), N is the number of voxel of coronary
    ix = ix[jx, jy, jz]                     # (ix, iy, iz) is the coordinate of the nearest coordinate of LV for each voxel of coronary, for now ix.shape = (N,)
    iy = iy[jx, jy, jz]
    iz = iz[jx, jy, jz]
    dist = cp.exp2(- dist[jx, jy, jz] / 70)

    dvf_list: list[torch.Tensor] = []
    zoom_rate = (1 / roi.get_zoom_rate()).flatten().tolist()
    for phase in tqdm(range(NUM_TOTAL_PHASE), desc='loading DVFs'):
        dvf_path = dvf_dir / f'phase_{phase:02d}.nii.gz'
        if not dvf_path.exists():
            raise FileNotFoundError(f"DVF file not found: {dvf_path}")
        dvf = nib.loadsave.load(dvf_path)
        dvf_data = dvf.get_fdata()  # type: ignore
        dvf_tensor = torch.from_numpy(dvf_data).to(device).half()
        dvf_tensor = dvf_tensor.squeeze().permute(3, 0, 1, 2)[None] # (1,3,H,W,D)
        dvf_tensor = F.interpolate(dvf_tensor, scale_factor=zoom_rate, mode='trilinear', align_corners=False)
        for i in range(3):
            dvf_tensor[:, i] = dvf_tensor[:, i] * zoom_rate[i]
        dvf_cp = cp.from_dlpack(to_dlpack(dvf_tensor))
        dvf_cp[:, :, jx, jy, jz] = dvf_cp[:, :, ix, iy, iz] * dist + dvf_cp[:, :, jx, jy, jz] * (1 - dist)
        dvf_tensor = from_dlpack(dvf_cp.toDlpack())
        dvf_list.append(dvf_tensor)

    del cavity, lv_cp, coronary_cp, dvf_cp
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

    if len(dvf_list) != NUM_TOTAL_PHASE:
        raise ValueError(f"Expected {NUM_TOTAL_PHASE} DVF files, found {len(dvf_list)}")

    drr_projections = []
    image_slicer_list = []
    image = image[None, None].half()
    coronary = coronary[None, None].half()
    for frame in tqdm(range(total_frame), desc='generating DRR projections'):
        d_alpha = get_delta_degree_at_frame(frame, fps=fps)
        d_phase = get_delta_phase_at_frame(frame, fps=fps)

        ddf = dvf2ddf(get_dvf_at_phase(dvf_list, d_phase))  # Shape (1,3,H,W,D)# (H,W,D,3) -> (3,H,W,D)
        warped_image = warp_image(image, ddf)
        warped_coronary = warp_coronary(coronary, ddf)
        image_slicer_list.append(warped_image.squeeze().cpu().numpy()[:, :, warped_image.shape[-1]//2])
    
        # Generate DRR projection
        drr_res = get_drr(warped_image, warped_coronary, image_affine, (alpha_start - d_alpha, beta_start, 0), mean_Hu_at_coronary, device)
        drr_res_np = drr_res.squeeze().detach().cpu().numpy().transpose(1, 0)
        drr_res_np = np.flip(drr_res_np, axis=0)
        torch.cuda.empty_cache()
        drr_projections.append(drr_res_np)
    
    drr_np = np.stack(drr_projections)
    drr_np = ((drr_np - drr_np.min()) / (drr_np.max() - drr_np.min()))*255
    drr_np = (255 - drr_np).astype(np.uint8)
    save_frames(output_dsa_path, drr_np, fps)
    slicers_np = np.stack(image_slicer_list).astype(np.float32)
    window_low, window_high = -300, 600
    slicers_np = np.clip(slicers_np, window_low, window_high)
    slicers_np = (slicers_np - window_low) / (window_high - window_low) * 255
    save_frames(output_dsa_path.parent / f'{output_dsa_path.stem}_slicers.tif', slicers_np.astype(np.uint8), fps)


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
        parser.add_argument('--cavity_path', type=Path, required=True,
                          help='Path to cavity segmentation')
        parser.add_argument('--output_path', type=Path, required=True,
                          help='Output TIFF file path')
        parser.add_argument('--total_frames', type=int, default=120,
                          help='Total frames to generate (default: 120)')
        parser.add_argument('--fps', type=float, default=60.0)
        
        args = parser.parse_args()
        
        print(f"Generating rotating DSA with {args.total_frames} frames...")
        generate_rotate_dsa(
            dvf_dir=args.dvf_dir,
            roi_json=args.roi_json,
            image_path=args.image_path,
            coronary_path=args.coronary_path,
            cavity_path=args.cavity_path,
            output_dsa_path=args.output_path,
            total_frame=args.total_frames,
            fps=args.fps
        )
        print("Done!")
    
    test_generate_rotate_dsa()
