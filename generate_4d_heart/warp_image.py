from pathlib import Path
from math import floor

from tqdm import tqdm
import torch
import nibabel as nib
import numpy as np
from monai.networks.blocks.warp import DVF2DDF, Warp
import numpy as np

from . import NUM_TOTAL_PHASE, LV_LABEL
from .roi import ROI
from .saver import save_nii
from .rotate_dsa.data_reader.data_reader import load_and_zoom_dvf

@torch.no_grad()
def generate_4d_cta(
    dvf_dir: Path,
    roi_json: Path,
    image_path: Path,
    coronary_path: Path,
    output_cta_path: Path,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dvf2ddf = DVF2DDF(num_steps=7, mode="bilinear", padding_mode="zeros").to(device)
    warp_image = Warp(mode="bilinear", padding_mode="zeros").to(device)
    
    roi = ROI.from_json(roi_json)
    
    def load_tensor(_path: Path) -> tuple[torch.Tensor, np.ndarray]:
        img = roi.crop(nib.loadsave.load(_path))  # type: ignore    # TODO: ROI 可能需要更好地应用，现在有点乱
        tensor = torch.from_numpy(img.get_fdata()).to(device).float()
        assert img.affine is not None
        return tensor, img.affine

    # TODO 可能需要加载原有影像并将裁剪后图像叠加显示
    image, image_affine = load_tensor(image_path)
    coronary, _ = load_tensor(coronary_path)
    
    image = image[None, None].half()
    coronary = coronary[None, None].half()
    
    for phase in tqdm(range(NUM_TOTAL_PHASE), desc=f'generating 4d cta for {image_path.name}...'):
        dvf_tensor = load_and_zoom_dvf(dvf_dir / f'phase_{phase:02d}.nii.gz', roi, device)
        ddf = dvf2ddf(dvf_tensor)
        save_nii(
            output_cta_path / "warped_image" / f"{phase:02d}.nii.gz",
            warp_image(image, ddf),
            image_affine,
            is_label=False
        )
        
        save_nii(
            output_cta_path / "warped_coronary" / f"{phase:02d}.nii.gz",
            warp_image(image, ddf),
            image_affine,
            is_label=True
        )