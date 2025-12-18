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
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib

matplotlib.use("Agg")


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
    frames: torch.Tensor | np.ndarray
) -> None:
    frames = frames.squeeze()
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, frames_np, imagej=True)

def save_png(
    output_path: Path,
    image_2d: torch.Tensor | np.ndarray
) -> None:
    image_2d = image_2d.squeeze()
    image_2d_np = image_2d.cpu().numpy() if isinstance(image_2d, torch.Tensor) else image_2d
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_2d_np)

def save_pngs(
    output_dir: Path,
    frames: torch.Tensor | np.ndarray
):
    frames = frames.squeeze()
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for t, image in tqdm(enumerate(frames_np), desc="Saving PNGs..."):
        if image.ndim == 2:
            image = image[..., None].repeat(3, axis=-1)     # convert to RGB format
        iio.imwrite(
            uri=output_dir / f"{t:03d}.png",
            image=image,
            plugin="pillow",
            extension=".png"
        )
    

def save_gif(
    output_path: Path,
    frames: torch.Tensor | np.ndarray,
    fps_gif: int = 30,
    **imshow_kwargs
) -> None:
    frames = frames.squeeze()
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames

    h, w = frames_np.shape[1], frames_np.shape[2]

    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes((0, 0, 1, 1))  # 填满整个 figure
    ax.axis("off")

    ims = []
    for i in range(frames_np.shape[0]):
        im = ax.imshow(frames_np[i], animated=True, **imshow_kwargs)
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig,
        ims,
        interval=1000 / fps_gif,
        blit=True,
        repeat_delay=1000
    )

    writer = animation.PillowWriter(fps=fps_gif)
    ani.save(
        output_path,
        writer=writer,
        dpi=dpi,
        savefig_kwargs={
            "pad_inches": 0
        }
    )

    plt.close(fig)


def save_deepthmap_gif(
    output_path: Path,
    depth_maps: torch.Tensor | np.ndarray,
    fps_gif: int = 30
) -> None:
    depth_maps = depth_maps.squeeze()
    depth_maps_np = depth_maps.cpu().numpy() if isinstance(depth_maps, torch.Tensor) else depth_maps
    vmin = np.min(depth_maps_np[depth_maps_np>0])
    vmax = np.max(depth_maps_np)
    
    save_gif(
        output_path,
        depth_maps_np,
        fps_gif=fps_gif,
        cmap='gray',
        vmin=vmin,
        vmax=vmax
    )