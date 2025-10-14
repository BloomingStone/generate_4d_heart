from pathlib import Path
import tifffile

from einops import rearrange
import torch
import numpy as np
import cv2
from tqdm import tqdm
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save as nib_save


def save_nii(
    output_path: Path,
    tensor: torch.Tensor, 
    affine: np.ndarray,
    is_label: bool = False
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.squeeze()
    if tensor.dim() == 4 and tensor.shape[0] == 3:
        tensor = rearrange(tensor, "c h w d -> h w d 1 c")
    elif tensor.dim() == 3:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    
    if is_label:
        image = Nifti1Image(tensor.cpu().numpy().astype(np.int8), affine)
    else:
        image = Nifti1Image(tensor.cpu().numpy().astype(np.float32), affine)
    nib_save(image, output_path)

def save_tif(
    output_path: Path,
    frames: torch.Tensor
) -> None:
    frames = frames.squeeze()
    T, W, H = frames.shape
    frames_np = frames.cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, frames_np, imagej=True)

def save_png(
    output_path: Path,
    image_2d: torch.Tensor
) -> None:
    image_2d = image_2d.squeeze()
    assert image_2d.dim() == 2
    image_2d_np = image_2d.cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_2d_np)


def save_mp4(
    output_path: Path,
    frames: torch.Tensor,
    fps: float
) -> None:
    frames = frames.squeeze()
    T, W, H = frames.shape
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (H, W), isColor=False)
    frames_np = frames.cpu().numpy()
    for frame in tqdm(frames_np, desc="Saving mp4"):
        video_writer.write(frame)
    video_writer.release()