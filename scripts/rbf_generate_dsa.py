from pathlib import Path
from typing import Literal
import logging
import hashlib
from enum import StrEnum

import numpy as np

from generate_4d_heart.rotate_dsa.contrast_simulator import StaticIodineContrast, FlowContrast
from generate_4d_heart.rotate_dsa.data_reader import RBFReader, StaticVolumeReader, DataReader, DataReaderResult
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters
from generate_4d_heart.rotate_dsa import RotateDSA
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.saver import save_nii

from cyclopts import App

app = App()


logging.basicConfig(
    filename='processing_errors.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] %(filename)s:%(lineno)d -- %(message)s',
    encoding='utf-8'
)

class OutputMode2D(StrEnum):
    DRR = "drr"
    ONLY_LABEL = "only-label"
    ONLY_INFO = "only-info"
    NULL = "null"

class OutputMode3D(StrEnum):
    COR_LABEL = "cor-label"
    COR_MESH = "cor-mesh"
    CAVITY_LABEL = "cavity-label"
    VOLUME = "volume"

class BatchDVFToDSA:
    def __init__(
        self,
        image_dir: Path,
        coronary_dir: Path,
        cavity_dir: Path,
        output_root: Path,
        dataset_name: str,
        run_random: bool = True,
        skip_existing: bool = False,
        output_mode_2d: OutputMode2D = OutputMode2D.DRR,
        output_mode_3d: tuple[DataReaderResult.SupportedSaveType, ...] = ("coronary", "cor-mesh", "cor-artifacts", "central-line", "meta"),
        output_phase_3d: float = 0.0,
        output_phase_3d_num: int = 1,
        reader_type: Literal["rbf", "static"] = "rbf",
        contrast_type: Literal["static", "flow"] = "static",
        dryrun: bool = False,
    ): 
            
        self.image_dir = image_dir
        self.coronary_dir = coronary_dir
        self.cavity_dir = cavity_dir
        self.output_root = output_root
        
        self.dataset_name = dataset_name
        self.run_random = run_random
        self.output_mode_2d = OutputMode2D(output_mode_2d)
        self.output_mode_3d = output_mode_3d
        self.output_phase_3d = CardiacPhase.from_time(output_phase_3d, 1.0)
        self.otuput_phase_3d_str = self.output_phase_3d.to_str(precision=2, has_decimal_point=False)
        self.output_phase_3d_num = output_phase_3d_num
        assert self.output_phase_3d_num >= 1, "output_phase_3d_num should be greater than or equal to 1"

        dirs = (image_dir, coronary_dir, cavity_dir)
        for d in dirs:
            assert d.exists(), f"{d} not exists"
            assert d.is_dir(), f"{d} is not a dir"
        
        self.paths_list: list[dict[str, Path]] = []
        for image_nii, coronary_nii, cavity_nii in zip(
            sorted(self.image_dir.iterdir()),
            sorted(self.coronary_dir.iterdir()),
            sorted(self.cavity_dir.iterdir())
        ):
            self.paths_list.append({
                "image_nii": image_nii,
                "coronary_nii": coronary_nii,
                "cavity_nii": cavity_nii
            })
        
        self.reader_type = reader_type
        self.skip_existing = skip_existing
        self.dryrun = dryrun
        self.torch_drr = TorchDRR(rotate_cfg=self._get_rotate_param())
        if contrast_type == "static":
            self.constrast_simulator = StaticIodineContrast()
        elif contrast_type == "flow":
            self.constrast_simulator = FlowContrast()
        else:
            raise ValueError(f"contrast_type {contrast_type} not supported")
        
    
    def _get_rotate_param(self):
        if self.run_random:
            rot_params = RotatedParameters.from_alpha_range(
                total_frame=180,
                alpha_start=-120.0,
                alpha_end=120.0,
                fps=60.0,
                beta_start=0.0,
                coordinate_system="RAS",
                parameterization="euler_angles",
                convention="ZXY"
            )
        else:
            rot_params = RotatedParameters()
        
        return rot_params

    def _output_3d(self, reader: DataReader, case_name: str, coronary_type: Literal["LCA", "RCA"], output_case_dir: Path):
        if self.output_mode_3d == ():
            return
        
        data_3d = reader.get_dynamic_data(self.output_phase_3d, coronary_type)
        
        data_3d.save(output_case_dir/self.otuput_phase_3d_str , save_types=self.output_mode_3d, coordinate_system="centering")

    
    def _output_2d(self, reader: DataReader, case_name: str, coronary_type: Literal["LCA", "RCA"], output_case_dir: Path):
        if self.output_mode_2d == "null":
            return
        
        dsa = RotateDSA(reader=reader, drr=self.torch_drr)
        
        match self.output_mode_2d:
            # donnot save phase_0_data here as there is _output_3d_one_phase() which will save phase_0_data if output_phase_3d_num > 1
            case OutputMode2D.DRR:
                dsa.run_and_save(output_case_dir, coronary_type, phase_0_save_types=())
            case OutputMode2D.ONLY_LABEL:
                dsa.run_and_save_no_drr(output_case_dir, coronary_type, phase_0_save_types=())
            case OutputMode2D.ONLY_INFO:
                dsa.save_no_run(output_case_dir, coronary_type, phase_0_save_types=())
            case _:
                raise ValueError(f"out_type_2d {self.output_mode_2d} not supported")

    def _gen_dsa_inner(
        self,
        reader: DataReader,
        case_name: str,
        coronary_type: Literal["LCA", "RCA"]
    ):
        case_name = f"{self.dataset_name}__{case_name}__{coronary_type}"
        output_case_dir = self.output_root / case_name
        output_case_dir.mkdir(parents=True, exist_ok=True)
        self._output_3d(reader, case_name, coronary_type, output_case_dir)
        self._output_2d(reader, case_name, coronary_type, output_case_dir)
            

    def run(self):
        for index, p in enumerate(self.paths_list):
            print(f"{index}/{len(self.paths_list)}")
            for p_item in p.values():
                print(p_item)
            
            image_nii = p["image_nii"]
            case_name = image_nii.stem.split('.')[0]

            if self.dryrun:
                for coronary_type in ("LCA", "RCA"):
                    if self.skip_existing:
                        output_case_dir = self.output_root / f"{self.dataset_name}__{case_name}__{coronary_type}"
                        if output_case_dir.exists() and any(output_case_dir.iterdir()):
                            print(f"[DRYRUN] (skip existing) {case_name} {coronary_type}")
                            continue
                    print(f"[DRYRUN] {case_name} {coronary_type}")
                continue
            
            if self.skip_existing:
                lca_dir = self.output_root / f"{self.dataset_name}__{case_name}__LCA"
                rca_dir = self.output_root / f"{self.dataset_name}__{case_name}__RCA"
                lca_exists = lca_dir.exists() and any(lca_dir.iterdir())
                rca_exists = rca_dir.exists() and any(rca_dir.iterdir())
                if lca_exists and rca_exists:
                    logging.info(f"Skipping existing case {case_name} (both LCA and RCA already exist)")
                    print(f"Skipping existing case {case_name} (both LCA and RCA already exist)")
                    continue
            
            try:
                logging.info(f"Generating reader:\n -volume: {p['image_nii']}\n -coronary: {p['coronary_nii']}\n -cavity: {p['cavity_nii']}")
                if self.reader_type == "rbf":
                    reader = RBFReader(
                        volume_nii=p["image_nii"],
                        cavity_nii=p["cavity_nii"],
                        coronary_nii=p["coronary_nii"],
                        contrast_simulator=self.constrast_simulator,
                    )
                elif self.reader_type == "static":
                    reader = StaticVolumeReader(
                        volume_path=p["image_nii"],
                        cavity_path=p["cavity_nii"],
                        coronary_path=p["coronary_nii"],
                        contrast_simulator=self.constrast_simulator
                    )
                else:
                    raise ValueError(f"reader_type {self.reader_type} not supported")
            except Exception as e:
                print(f"Error processing {image_nii}: {e}")
                logging.exception(f"{image_nii} failed due to error: {e}")
                continue
            
            for coronary_type in ("LCA", "RCA"):
                
                if self.skip_existing:
                    output_case_dir = self.output_root / f"{self.dataset_name}__{case_name}__{coronary_type}"
                    if output_case_dir.exists() and any(output_case_dir.iterdir()):
                        logging.info(f"Skipping existing case {case_name} coronary {coronary_type} (output dir not empty)")
                        print(f"Skipping existing case {case_name} coronary {coronary_type}")
                        continue
                
                try:
                    logging.info(f"Processing case {case_name} coronary {coronary_type}")
                    self._gen_dsa_inner(reader, case_name, coronary_type)
                except Exception as e:
                    print(f"Error processing {image_nii} with {coronary_type}: {e}")
                    logging.exception(f"{image_nii} with {coronary_type} failed due to error: {e}")
                    continue

@app.default()
def main(
    dataset_names: list[Literal["asoca-normal", "asoca-diseased", "shanghai"]],
    output_root: Path,
    random_seed: int = 42,
    use_random_seed: bool = True,
    output_mode_2d: OutputMode2D = OutputMode2D.DRR,
    output_mode_3d: tuple[DataReaderResult.SupportedSaveType, ...] = ("coronary", "cor-mesh", "cor-artifacts", "central-line", "meta"),
    output_phase_3d: float = 0.0,
    output_phase_3d_num: int = 1,
    reader_type: Literal["rbf", "static"] = "rbf",
    contrast_type: Literal["static", "flow"] = "static",
    skip_existing: bool = False,
    dryrun: bool = False,
):
    """
    example command:
    python scripts/batch_rbf.py --dataset-names shanghai --contrast-type flow  --output_root /media/data3/sj/Data/gen4d_outputs_flow --dryrun --skip-existing
    
    Parameters:
    ____
    dataset_names: list of dataset names to process, currently supports "asoca-normal" and "asoca-diseased"
    output_root: root directory to save the output data
    random_seed: random seed for reproducibility, default is 42
    use_random_seed: whether to use random seed, if False, the output will be the same for each run, default is True
    output_mode_2d: output mode for 2D data, can be "drr", "only-label", "only-info" or "null", default is "drr"
    output_mode_3d: list of output modes for 3D data, can be "cor-label", "cor-mesh", "cavity-label" or "volume", default is ["cor-label", "cor-mesh"]
    output_phase_3d: cardiac phase to output 3D data, should be between 0 and 1, default is 0.0 (end-diastole)
    output_phase_3d_num: number of 3D phases to output, default is 1. If greater than 1, it will output multiple phases evenly spaced between 0 and 1. If output_phase_3d_num > 1, the output_phase_3d parameter will be ignored. (Not Implemented yet)
    contrast_type: type of contrast simulation to use, can be "static" or "flow", default is "static"
    skip_existing: if True, skip cases whose output directory already exists and is not empty, default is False
    dryrun: if True, only print the cases that would be processed without actually running, default is False
    """
    for d in dataset_names:
        d = d.lower()
        
        if use_random_seed:
            name_hash = int(hashlib.sha256(d.lower().encode()).hexdigest(), 16)
            combined_seed = (name_hash + random_seed) % (2**32)  # numpy seed 必须 < 2**32
            np.random.seed(combined_seed)
    
        match d:
            case "asoca-normal":
                print(f"{d} -- {output_root} -- seed:{random_seed}")
                origin_dir = Path("/media/data3/sj/Data/ASOCA/Normal/")
                image_dir=origin_dir / "CTCA_nii"
                coronary_dir=origin_dir / "Annotations_nii"
                cavity_dir=origin_dir / "cavity"
            
            case "asoca-diseased":
                print(f"{d} -- {output_root} -- seed:{random_seed}")
                origin_dir = Path("/media/data3/sj/Data/ASOCA/Diseased_partial")
                image_dir=origin_dir / "CTCA_nii"
                coronary_dir=origin_dir / "coronary"
                cavity_dir=origin_dir / "cavity"

            case "shanghai":
                print(f"{d} -- {output_root} -- seed:{random_seed}")
                origin_dir = Path("/media/data2/sj/Data/Shanghai_139_partial")
                image_dir=origin_dir / "img"
                coronary_dir=origin_dir / "seg"
                cavity_dir=origin_dir / "cavity"
            
            case _:
                print(f"{d} not supported")
                raise ValueError(f"{d} not supported")
        
        dsa = BatchDVFToDSA(
            image_dir=image_dir,
            coronary_dir=coronary_dir,
            cavity_dir=cavity_dir,
            output_root=output_root,
            dataset_name=d,
            run_random=use_random_seed,
            skip_existing=skip_existing,
            dryrun=dryrun,
            output_mode_2d=output_mode_2d,
            output_mode_3d=output_mode_3d,
            output_phase_3d=output_phase_3d,
            output_phase_3d_num=output_phase_3d_num,
            reader_type=reader_type,
            contrast_type=contrast_type
        )
    
        dsa.run()

if __name__ == "__main__":
    app()
