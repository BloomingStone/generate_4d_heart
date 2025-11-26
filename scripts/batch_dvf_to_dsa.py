from pathlib import Path
from typing import Literal
import logging
import hashlib

import numpy as np

from generate_4d_heart.rotate_dsa.movement_enhancer import CoronaryBoundLVLinear
from generate_4d_heart.rotate_dsa.contrast_simulator import MultipliContrast
from generate_4d_heart.rotate_dsa.data_reader import VolumeDVFReader
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters
from generate_4d_heart.rotate_dsa import RotateDSA


logging.basicConfig(
    filename='processing_errors.log',
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - [%(levelname)s] %(filename)s:%(lineno)d -- %(message)s',
    encoding='utf-8'
)

class BatchDVFToDSA:
    def __init__(
        self,
        gen_dvf_output_dir: Path,
        image_dir: Path,
        coronary_dir: Path,
        cavity_dir: Path,
        output_root: Path,
        dataset_name: str,
        recover_cropped_data: bool = True,
        run_random: bool = False,
        do_enhance: bool = True,
        only_output_label = False
    ):
        self.gen_dvf_output_dir = gen_dvf_output_dir
        self.dvf_dir = gen_dvf_output_dir / "dvf"
        self.roi_dir = gen_dvf_output_dir / "roi_info"
        
        self.image_dir = image_dir
        self.coronary_dir = coronary_dir
        self.cavity_dir = cavity_dir
        self.output_root = output_root
        
        self.dataset_name = dataset_name
        self.recover_cropped_data = recover_cropped_data
        self.run_random = run_random
        self.do_enhance = do_enhance
        self.only_output_label = only_output_label
        
        dirs = (self.dvf_dir, self.roi_dir, image_dir, coronary_dir, cavity_dir)
        for d in dirs:
            assert d.exists(), f"{d} not exists"
            assert d.is_dir(), f"{d} is not a dir"
        
        self.paths_list: list[dict[str, Path]] = []
        for sub_dvf_dir, roi_json, image_nii, coronary_nii, cavity_nii in zip(
            sorted(self.dvf_dir.iterdir()),
            sorted(self.roi_dir.iterdir()),
            sorted(self.image_dir.iterdir()),
            sorted(self.coronary_dir.iterdir()),
            sorted(self.cavity_dir.iterdir())
        ):
            assert sub_dvf_dir.is_dir()
            self.paths_list.append({
                "sub_dvf_dir": sub_dvf_dir,
                "roi_json": roi_json,
                "image_nii": image_nii,
                "coronary_nii": coronary_nii,
                "cavity_nii": cavity_nii
            })
        
    
    def _get_rotate_param(self):
        if self.run_random:
            rot_params = RotatedParameters(
                total_frame=np.random.randint(80, 100),         # 80 ~ 100
                alpha_start=np.random.rand() * 360,             # 0 ~ 360
                angular_velocity=np.random.rand() * 10 + 70,    # 70 ~ 80
                beta_start=np.random.randn() * 10               # N(0, 10)
            )
        else:
            rot_params = RotatedParameters()
        
        return rot_params

    def _gen_dsa_inner(
        self,
        reader: VolumeDVFReader,
        case_name: str,
        coronary_type: Literal["LCA", "RCA"]
    ):
        case_name = f"{self.dataset_name}__{case_name}__{coronary_type}"
        output_case_dir = self.output_root / case_name
        output_case_dir.mkdir(parents=True, exist_ok=True)

        dsa = RotateDSA(
            reader, MultipliContrast(), 
            TorchDRR(rotate_cfg=self._get_rotate_param())
        )
        
        if self.only_output_label:
            dsa.run_and_save_no_drr(output_case_dir, coronary_type)
        else:
            dsa.run_and_save(output_case_dir, coronary_type)

    def run(self):
        for index, p in enumerate(self.paths_list):
            print(f"{index}/{len(self.paths_list)}")
            for p_item in p.values():
                print(p_item)
            
            image_nii = p["image_nii"]
            case_name = image_nii.stem.split('.')[0]
            
            if self.do_enhance:
                enhancer = CoronaryBoundLVLinear
            else:
                enhancer = None
            
            try:
                reader = VolumeDVFReader(
                    p["image_nii"], p["cavity_nii"], p["coronary_nii"], 
                    p["roi_json"], p["sub_dvf_dir"], enhancer,
                    recover_cropped_data=self.recover_cropped_data
                )
            except Exception as e:
                logging.exception(f"{image_nii} failed due to error: {e}")
                continue
            
            for coronary_type in ("LCA", "RCA"):
                
                try:
                    self._gen_dsa_inner(reader, case_name, coronary_type)
                except Exception as e:
                    logging.exception(f"{image_nii} with {coronary_type} failed due to error: {e}")
                    continue

def main(
    dataset_names: list[str],
    output_root: Path,
    random_seed: int|None=None,
    output_only_label: bool=False
):
    for d in dataset_names:
        d = d.lower()
        
        if random_seed is not None:
            name_hash = int(hashlib.sha256(d.lower().encode()).hexdigest(), 16)
            combined_seed = (name_hash + random_seed) % (2**32)  # numpy seed 必须 < 2**32
            np.random.seed(combined_seed)
            run_random = True
        else:
            run_random = False
        
        if d not in ("asoca", "shanghai", "imagecas"):
            raise ValueError(f"{d} is not supported")
    
        match d:
            case "asoca":
                print(f"{d} -- {output_root} -- seed:{random_seed}")
                origin_dir = Path("/media/data3/sj/Data/ASOCA/normal_gen_4d")
                dsa = BatchDVFToDSA(
                    gen_dvf_output_dir=Path("/media/data3/sj/Data/ASOCA/normal_gen_4d_output_new"),
                    image_dir=origin_dir/"CTCA_nii",
                    coronary_dir=origin_dir/"coronary",
                    cavity_dir=origin_dir/"cavity",
                    output_root=output_root,
                    dataset_name=d,
                    run_random=run_random,
                    only_output_label=output_only_label
                )
            case "shanghai":
                print(f"{d} -- {output_root} -- seed:{random_seed}")
                origin_dir = Path("/media/data3/sj/Data/Shanghai_139_partial")
                dsa = BatchDVFToDSA(
                    gen_dvf_output_dir=Path("/media/data3/sj/Data/Shanghai_139_partial/4d_heart"),
                    image_dir=origin_dir/"re_affined_image",
                    coronary_dir=origin_dir/"re_affined_coronary",
                    cavity_dir=origin_dir/"cavity",
                    output_root=output_root,
                    dataset_name=d,
                    # recover_cropped_data=False,     # 原图尺寸太大，如果复原会导致超显存(改用 compied drr 后解决)
                    run_random=run_random,
                    only_output_label=output_only_label
                )
            case "imagecas":
                print(f"{d} -- {output_root} -- seed:{random_seed}")
                origin_dir = Path("/media/data3/sj/Data/imageCAS")
                dsa = BatchDVFToDSA(
                    gen_dvf_output_dir=Path("/media/data3/sj/Data/imageCAS/gen_4d_output"),
                    image_dir=origin_dir/"imageCAS_Selected_nnunet",
                    coronary_dir=origin_dir/"coronary",
                    cavity_dir=origin_dir/"postprocessed",
                    output_root=output_root,
                    dataset_name=d,
                    # recover_cropped_data=False,     # 原图尺寸太大，如果复原会导致超显存
                    run_random=run_random,
                    only_output_label=output_only_label
                )
            case _:
                print(f"{d} not supported")
                raise ValueError(f"{d} not supported")
    
        dsa.run()

if __name__ == "__main__":
    import typer
    typer.run(main)
