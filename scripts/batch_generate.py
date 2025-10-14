from pathlib import Path
import os

from generate_4d_heart.warp_image import generate_4d_cta

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

gen_dvf_output_dir = Path("/media/data3/sj/Data/ASOCA/normal_gen_4d_output_new")
dvf_dir = gen_dvf_output_dir / "dvf"
roi_dir = gen_dvf_output_dir / "roi_info"

origin_dir = Path("/media/data3/sj/Data/ASOCA/normal_gen_4d")
image_dir = origin_dir / "CTCA_nii"
coronary_dir = origin_dir / "coronary"

output_root = Path("/media/data3/sj/Data/ASOCA/4d_heart")


for sub_dvf_dir, roi_json, image_nii, coronary_nii in zip(
    sorted(dvf_dir.iterdir()),
    sorted(roi_dir.iterdir()),
    sorted(image_dir.iterdir()),
    sorted(coronary_dir.iterdir())
):
    output_dir = output_root / image_nii.name.split('.')[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_4d_cta(
        sub_dvf_dir, roi_json, image_nii, coronary_nii, output_dir
    )
