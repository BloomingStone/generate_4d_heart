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
    print(f"saving: {output_path}")
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
    print(f"saving: {output_path}")
    frames = frames.squeeze()
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    if frames_np.ndim == 2:
        frames_np = frames_np[None]  # (H, W) → (1, H, W)
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
    print(f"saving pngs to: {output_dir}")
    frames = frames.squeeze()
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    if frames_np.ndim == 2:
        frames_np = frames_np[None]  # (H, W) → (1, H, W)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for t, image in enumerate(frames_np):
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
    fps: int = 30,
    vmin: float | None = None,
    vmax: float | None = None,
    **imshow_kwargs
) -> None:
    print(f"saving: {output_path}")
    frames = frames.squeeze()
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    if frames_np.ndim == 2:
        frames_np = frames_np[None]  # (H, W) → (1, H, W)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames_np.shape[1], frames_np.shape[2]

    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes((0, 0, 1, 1))  # 填满整个 figure
    ax.axis("off")
    
    if vmin is None:
        vmin = np.min(frames_np)
    if vmax is None:
        vmax = np.max(frames_np)

    ims = []
    for i in range(frames_np.shape[0]):
        im = ax.imshow(frames_np[i], animated=True, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig,
        ims,
        interval=1000 / fps,
        blit=True,
        repeat_delay=1000
    )

    writer = animation.PillowWriter(fps=fps)
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
    fps: int = 30,
    cmap: str = 'gray'
) -> None:
    depth_maps = depth_maps.squeeze()
    depth_maps_np = depth_maps.cpu().numpy() if isinstance(depth_maps, torch.Tensor) else depth_maps
    pos = depth_maps_np[depth_maps_np > 0]
    vmin = np.min(pos) if pos.size > 0 else 0.0
    vmax = np.max(depth_maps_np)
    
    save_gif(
        output_path,
        depth_maps_np,
        fps=fps,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )


def save_video(
    output_path: Path,
    frames: torch.Tensor | np.ndarray,
    fps: float = 30.0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save frames as an MP4 video using imageio-ffmpeg (H.264).

    Frames are normalized to [0, 255] uint8 using the provided vmin/vmax
    (or auto-scaled from data min/max if not given).
    """
    print(f"saving: {output_path}")
    frames = frames.squeeze()
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    if frames_np.ndim == 2:
        frames_np = frames_np[None]  # (H, W) → (1, H, W)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    vmin_ = np.min(frames_np) if vmin is None else vmin
    vmax_ = np.max(frames_np) if vmax is None else vmax

    # Normalize to [0, 255] uint8
    if vmax_ - vmin_ > 1e-8:
        frames_uint8 = np.clip((frames_np - vmin_) / (vmax_ - vmin_) * 255, 0, 255).astype(np.uint8)
    else:
        frames_uint8 = np.zeros_like(frames_np, dtype=np.uint8)

    # Convert to 3-channel RGB for compatibility
    if frames_uint8.ndim == 3:
        frames_uint8 = frames_uint8[..., None].repeat(3, axis=-1)

    # Write with imageio (uses ffmpeg backend, H.264 codec, compatible with VS Code)
    iio.imwrite(output_path, frames_uint8, fps=fps, codec='libx264')


def save_deepthmap_video(
    output_path: Path,
    depth_maps: torch.Tensor | np.ndarray,
    fps: float = 30.0,
) -> None:
    depth_maps = depth_maps.squeeze()
    depth_maps_np = depth_maps.cpu().numpy() if isinstance(depth_maps, torch.Tensor) else depth_maps
    pos = depth_maps_np[depth_maps_np > 0]
    vmin = np.min(pos) if pos.size > 0 else 0.0
    vmax = np.max(depth_maps_np)

    save_video(
        output_path,
        depth_maps_np,
        fps=fps,
        vmin=vmin,
        vmax=vmax,
    )