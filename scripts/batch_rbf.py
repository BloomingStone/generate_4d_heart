from pathlib import Path
from typing import Literal
import logging
import hashlib
from enum import StrEnum

import numpy as np

from generate_4d_heart.rotate_dsa.contrast_simulator import MultipliContrast
from generate_4d_heart.rotate_dsa.data_reader import RBFReader
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
        output_mode_2d: OutputMode2D = OutputMode2D.DRR,
        output_mode_3d: list[OutputMode3D]|None = None,
        output_phase_3d: float = 0.0,
        output_phase_3d_num: int = 1
    ): 
        if output_mode_3d is None:
            output_mode_3d = [OutputMode3D.COR_LABEL, OutputMode3D.COR_MESH]
            
        self.image_dir = image_dir
        self.coronary_dir = coronary_dir
        self.cavity_dir = cavity_dir
        self.output_root = output_root
        
        self.dataset_name = dataset_name
        self.run_random = run_random
        self.output_mode_2d = OutputMode2D(output_mode_2d)
        output_model_3d = set(OutputMode3D(m) for m in output_mode_3d)
        self.output_mode_3d = output_model_3d
        self.output_phase_3d = CardiacPhase(output_phase_3d)
        self.otuput_phase_3d_str = self.output_phase_3d.to_str(precision=2, has_decimal_point=False)
        assert self.output_phase_3d_num >= 1, "output_phase_3d_num should be greater than or equal to 1"
        self.output_phase_3d_num = output_phase_3d_num

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
        
        self.torch_drr = TorchDRR(rotate_cfg=self._get_rotate_param())
        self.constrast_simulator = MultipliContrast()
        
    
    def _get_rotate_param(self):
        if self.run_random:
            rot_params = RotatedParameters(
                total_frame=120,    # 120 frames / 30 fps = 4s video
                alpha_start=np.random.rand() * 180 - 90,        # -90 ~ 90
            )
        else:
            rot_params = RotatedParameters()
        
        return rot_params

    def _output_3d_one_phase(self, reader: RBFReader, case_name: str, coronary_type: Literal["LCA", "RCA"], output_case_dir: Path):
        if self.output_mode_3d == "null":
            return
        
        data_3d = reader.get_data(self.output_phase_3d, coronary_type)
        
        for mode in self.output_mode_3d:
            match mode:
                case OutputMode3D.VOLUME:
                    save_nii(
                        output_case_dir / f"{self.otuput_phase_3d_str}_image.nii.gz",
                        data_3d.volume,
                        affine=data_3d.affine,
                        is_label=False
                    )
                case OutputMode3D.CAVITY_LABEL:
                    save_nii(
                        output_case_dir / f"{self.otuput_phase_3d_str}_cavity_label.nii.gz",
                        data_3d.cavity_label,
                        affine=data_3d.affine,
                        is_label=True
                    )
                case OutputMode3D.COR_LABEL:
                     save_nii(
                        output_case_dir / f"{coronary_type}_{self.otuput_phase_3d_str}_label.nii.gz",
                        data_3d.coronary.label,
                        affine=data_3d.coronary.centering_affine,
                        is_label=True
                    )
                case OutputMode3D.COR_MESH:
                    data_3d.coronary.mesh_in_world.save(output_case_dir / f"{coronary_type}_{self.otuput_phase_3d_str}_mesh.vtk")
                case _:
                    raise ValueError(f"output_mode_3d {mode} not supported")

    
    def _output_2d(self, reader: RBFReader, case_name: str, coronary_type: Literal["LCA", "RCA"], output_case_dir: Path):
        if self.output_mode_2d == "null":
            return
        
        dsa = RotateDSA( reader, self.constrast_simulator, self.torch_drr )
        
        match self.output_mode_2d:
            case OutputMode2D.DRR:
                dsa.run_and_save(output_case_dir, coronary_type)
            case OutputMode2D.ONLY_LABEL:
                dsa.run_and_save_no_drr(output_case_dir, coronary_type)
            case OutputMode2D.ONLY_INFO:
                dsa.save_no_run(output_case_dir, coronary_type)
            case _:
                raise ValueError(f"out_type_2d {self.output_mode_2d} not supported")

    def _gen_dsa_inner(
        self,
        reader: RBFReader,
        case_name: str,
        coronary_type: Literal["LCA", "RCA"]
    ):
        case_name = f"{self.dataset_name}__{case_name}__{coronary_type}"
        output_case_dir = self.output_root / case_name
        output_case_dir.mkdir(parents=True, exist_ok=True)
        if self.output_phase_3d_num == 1:
            self._output_3d_one_phase(reader, case_name, coronary_type, output_case_dir)
        elif self.output_phase_3d_num > 1:
            # TODO
            raise NotImplementedError("output_phase_3d_num > 1 is not implemented yet")    
        else:
            raise ValueError(f"output_phase_3d_num should be greater than or equal to 1, but got {self.output_phase_3d_num}")
        self._output_2d(reader, case_name, coronary_type, output_case_dir)
            

    def run(self):
        for index, p in enumerate(self.paths_list):
            print(f"{index}/{len(self.paths_list)}")
            for p_item in p.values():
                print(p_item)
            
            image_nii = p["image_nii"]
            case_name = image_nii.stem.split('.')[0]
            
            try:
                reader = RBFReader(
                    p["image_nii"], p["cavity_nii"], p["coronary_nii"],
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

@app.default()
def main(
    dataset_names: list[Literal["asoca-normal", "asoca-diseased"]],
    output_root: Path,
    random_seed: int = 42,
    use_random_seed: bool = True,
    output_mode_2d: OutputMode2D = OutputMode2D.DRR,
    output_mode_3d: list[OutputMode3D] = [OutputMode3D.COR_LABEL, OutputMode3D.COR_MESH],
    output_phase_3d: float = 0.0,
    output_phase_3d_num: int = 1
):
    """
    
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
                origin_dir = Path("/media/E/sj/Data/ASOCA/Normal/normal_gen_4d")
                dsa = BatchDVFToDSA(
                    image_dir=origin_dir/"CTCA_nii",
                    coronary_dir=origin_dir/"coronary",
                    cavity_dir=origin_dir/"cavity",
                    output_root=output_root,
                    dataset_name=d,
                    run_random=use_random_seed,
                    output_mode_2d=output_mode_2d,
                    output_mode_3d=output_mode_3d,
                    output_phase_3d=output_phase_3d,
                    output_phase_3d_num=output_phase_3d_num
                )
            case "asoca-diseased":
                print(f"{d} -- {output_root} -- seed:{random_seed}")
                origin_dir = Path("/media/E/sj/Data/ASOCA/Diseased_partial")
                dsa = BatchDVFToDSA(
                    image_dir=origin_dir/"CTCA_nii",
                    coronary_dir=origin_dir/"coronary",
                    cavity_dir=origin_dir/"cavity",
                    output_root=output_root,
                    dataset_name=d,
                    run_random=use_random_seed,
                    output_mode_2d=output_mode_2d,
                    output_mode_3d=output_mode_3d,
                    output_phase_3d=output_phase_3d,
                    output_phase_3d_num=output_phase_3d_num
                )
            case _:
                print(f"{d} not supported")
                raise ValueError(f"{d} not supported")
    
        dsa.run()

if __name__ == "__main__":
    app()
