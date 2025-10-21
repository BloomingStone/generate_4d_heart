from pathlib import Path
import tifffile
import imageio.v3 as iio

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

def save_pngs(
    output_dir: Path,
    frames: torch.Tensor
):
    frames = frames.squeeze()
    frames_np = frames.cpu().numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for t, image in tqdm(enumerate(frames_np), desc="Saving PNGs..."):
        iio.imwrite(
            uri=output_dir / f"{t:03d}.png",
            image=image,
            plugin="pillow",
            extension=".png"
        )
    

def save_gif(
    output_path: Path,
    frames: torch.Tensor,
    fps: float
) -> None:
    frames = frames.squeeze()
    frames_np = frames.cpu().numpy()
    iio.imwrite(output_path, frames_np, extension=".gif", fps=fps)