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
from cupyx.scipy.ndimage import label, center_of_mass, distance_transform_edt, gaussian_filter, binary_dilation
from torch.utils.dlpack import from_dlpack, to_dlpack
from einops import rearrange

from . import NUM_TOTAL_PHASE, LV_LABEL
from .roi import ROI
import numpy as np
import imageio.v2 as imageio


def get_delta_degree_at_frame(frame: int, omega: float | int, fps: int | float) -> float:
    """
    Args:
        frame (int): the frame number, starting from 0
        omega (float | int): degree velocity, default 75 degree per second
        fps (int): frame per second, default 60 fps

    Returns:
        float: the degree of rotation at given frame
    """
    return frame * omega / fps

def get_delta_phase_at_frame(frame: int, fps: int | float) -> float:
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
    translations = torch.tensor([[0.0, 1400.0, 0.0]], device=device)

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
    

def save_nifti(tensor: torch.Tensor, affine: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.squeeze()
    if tensor.dim() == 4 and tensor.shape[0] == 3:
        tensor = rearrange(tensor, "c h w d -> h w d 1 c")
    elif tensor.dim() == 3:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    nib.save(nib.Nifti1Image(tensor.cpu().numpy().astype(np.float32), affine), output_path)

def read_and_zoom_dvf(dvf_path: Path, zoom_rate: tuple[float, float, float], device: torch.device) -> torch.Tensor:
    if not dvf_path.exists():
        raise FileNotFoundError(f"DVF file not found: {dvf_path}")
    dvf = nib.loadsave.load(dvf_path)
    dvf_data = dvf.get_fdata()  # type: ignore
    dvf_tensor = torch.from_numpy(dvf_data).to(device).half()
    dvf_tensor = dvf_tensor.squeeze().permute(3, 0, 1, 2)[None] # (1,3,H,W,D)
    dvf_tensor = F.interpolate(dvf_tensor, scale_factor=zoom_rate, mode='trilinear', align_corners=False)
    for i in range(3):
        dvf_tensor[:, i] = dvf_tensor[:, i] * zoom_rate[i]
    return dvf_tensor

# TODO seperate into classes DVF DRR and saver
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
    beta_start: float = 0.0,
    omega: float = 75.0,
    fps: float = 60.0,
    enhance_coronary_movement: bool = True,
    save_warped_image: bool = False,
    save_warped_coronary: bool = False,
    save_ddf: bool = False,
    save_dvf: bool = False,
):
    """
    Args:
        dvf_dir (Path): the directory of dvf of each phase
        roi_json (Path): the roi json file (contains affine, crop box and zoom factor)
        image_path (Path): the original image path
        coronary_path (Path): the original coronary mask path
        cavity_path (Path): the original cavity mask path
        output_dsa_path (Path): the output directory of rotated dsa
        total_frame (int): the total frame of rotated dsa
        alpha_start (float): the start angle of rotation (in degree, Z axis)
        beta_start (float): the start angle of rotation (in degree, Y axis)
        fps (float): the fps of the output video
        omega (float): the angle velocity of rotation (in degrees per second, Z axis), the angle of rotation. alpha = alpha_start + omega * t
        enhance_coronary_movement (bool): whether to enhance the movement of coronary
        save_warped_image (bool): whether to save the warped image, total frame = __module__.NUM_TOTAL_PHASE, output path = output_dsa_path / "warped_image", similarly hereinafter
        save_warped_coronary (bool): whether to save the warped coronary mask
        save_ddf (bool): whether to save the ddf
        save_dvf (bool): whether to save the dvf
    Returns:
        None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dvf2ddf = DVF2DDF(num_steps=7, mode="bilinear", padding_mode="zeros").to(device)
    warp_image = Warp(mode="bilinear", padding_mode="zeros").to(device)
    warp_coronary = Warp(mode="nearest", padding_mode="zeros").to(device)
    roi = ROI.from_json(roi_json)
    
    def load_tensor(_path: Path) -> torch.Tensor:
        img = roi.crop(nib.loadsave.load(_path))  # type: ignore    # TODO: ROI 可能需要更好地应用，现在有点乱
        tensor = torch.from_numpy(img.get_fdata()).to(device).float()
        return tensor, img.affine

    # TODO 可能需要加载原有影像并将裁剪后图像叠加显示
    image, image_affine = load_tensor(image_path)
    coronary, _ = load_tensor(coronary_path)
    cavity, _ = load_tensor(cavity_path)
    image_before_cropped = torch.from_numpy(nib.loadsave.load(image_path).get_fdata()).to(device).float()
    mean_Hu_at_coronary = image[coronary==1].mean().item()
    
    if enhance_coronary_movement:
        # use dvf at the LV part to enhance the movement of coronary
        lv_cp = cp.from_dlpack(to_dlpack(cavity==LV_LABEL)).astype(cp.bool_)  # (W,H,D), bool
        coronary_cp = cp.from_dlpack(to_dlpack(coronary)).astype(cp.bool_)  # (W,H,D), bool
        dilate_coronary_cp = binary_dilation(coronary_cp, iterations=1)
        dist, indices = distance_transform_edt(~lv_cp, return_distances=True, return_indices=True)  # for now ix.shape = (W,H,D)
        indices = indices[:, dilate_coronary_cp] # indices now is the coordinate of the nearest coordinate of LV for each voxel of coronary, for now ix.shape = (N,)
        # alpha = cp.exp2(- dist[dilate_coronary_cp] / 50)    # TODO need to explose this parameter
        alpha = cp.clip(1 - dist[dilate_coronary_cp] / 50, 0, 1)    # TODO need to explose this parameter

    image = image[None, None].half()
    coronary = coronary[None, None].half()
    def optional_save():
        f = lambda flag, x, name: save_nifti(x, image_affine, output_dsa_path.parent / name / f"{phase:02d}.nii.gz") if flag else None
        
        if not(save_warped_image or save_warped_coronary or save_ddf or save_dvf):
            return
        f(save_dvf, dvf_tensor, "dvf")
        
        if not(save_warped_image or save_warped_coronary or save_ddf):
            return
        ddf = dvf2ddf(dvf_tensor)
        f(save_ddf, ddf, "ddf")
        f(save_warped_image, warp_image(image, ddf), "warped_image")
        f(save_warped_coronary, warp_coronary(coronary, ddf), "warped_coronary")
    
    smooth_mask = binary_dilation(coronary_cp, structure=cp.ones((3,3,3)))
    dvf_list: list[torch.Tensor] = []
    zoom_rate = (1 / roi.get_zoom_rate()).flatten().tolist()    # image saved as spacing of 1mm, so we need to zoom it back to the original spacing
    for phase in tqdm(range(NUM_TOTAL_PHASE), desc='loading and updating DVFs'):
        dvf_tensor = read_and_zoom_dvf(dvf_dir / f'phase_{phase:02d}.nii.gz', zoom_rate, device)
        
        # TODO: 这里或许可以放到存储DVF的时候做？（但存储DVF时用的是裁剪并缩放过的，可能有影响）
        # TODO: 拆分成单独类或函数
        if enhance_coronary_movement:
            dvf_cp = cp.from_dlpack(to_dlpack(dvf_tensor))
            dvf_cp[:, :, dilate_coronary_cp] = dvf_cp[:, :, *indices] * alpha + dvf_cp[:, :, dilate_coronary_cp] * (1 - alpha)
            smoothed = cp.zeros_like(dvf_cp)
            for i in range(3):
                smoothed[:, i] = gaussian_filter(dvf_cp[:, i], sigma=2.0)
            dvf_cp[:, :, smooth_mask] = smoothed[:, :, smooth_mask]
            dvf_tensor = from_dlpack(dvf_cp.toDlpack())
        
        dvf_list.append(dvf_tensor)
        optional_save()

    del cavity, lv_cp, coronary_cp, dvf_cp
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

    drr_projections = []
    image_slicer_list = []
    for frame in tqdm(range(total_frame), desc='generating DRR projections'):
        d_alpha = get_delta_degree_at_frame(frame, omega=omega, fps=fps)
        d_phase = get_delta_phase_at_frame(frame, fps=fps)

        ddf = dvf2ddf(get_dvf_at_phase(dvf_list, d_phase))  # Shape (1,3,H,W,D)# (H,W,D,3) -> (3,H,W,D)
        warped_image = roi.recover_cropped_tensor(warp_image(image, ddf), image_before_cropped[None, None])
        warped_coronary = roi.recover_cropped_tensor(warp_coronary(coronary, ddf))
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
    # TODO 需要输出每一帧的投影几何参数，便于后续处理

if __name__ == '__main__':
    import typer
    typer.run(generate_rotate_dsa)