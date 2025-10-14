from pathlib import Path
import os

from generate_4d_heart.warp_image import generate_4d_cta

import logging

logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - [%(levelname)s] %(filename)s:%(lineno)d -- %(message)s',
    encoding='utf-8'
)

logger = logging.getLogger("batch-gen-cta")

gen_dvf_output_dir = Path("/media/data3/sj/Data/Shanghai_139_partial/4d_heart")
dvf_dir = gen_dvf_output_dir / "dvf"
roi_dir = gen_dvf_output_dir / "roi_info"

origin_dir = Path("/media/data3/sj/Data/Shanghai_139_partial")
image_dir = origin_dir / "re_affined_image"
coronary_dir = origin_dir / "re_affined_coronary"

output_root = Path("/media/data3/sj/Data/Shanghai_139_partial/4d_heart/cta")


for sub_dvf_dir, roi_json, image_nii, coronary_nii in zip(
    sorted(dvf_dir.iterdir()),
    sorted(roi_dir.iterdir()),
    sorted(image_dir.iterdir()),
    sorted(coronary_dir.iterdir())
):
    try:
        output_dir = output_root / image_nii.name.split('.')[0]
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_4d_cta(
            sub_dvf_dir, roi_json, image_nii, coronary_nii, output_dir
        )
    except Exception as e:
        logger.error(f"{dvf_dir.name} failed due to error: {e}")
        continue
