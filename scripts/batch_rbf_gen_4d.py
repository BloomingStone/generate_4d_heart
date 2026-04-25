from pathlib import Path
from typing import Literal
import logging
from enum import StrEnum
from queue import Queue
from threading import Thread, Lock

from tqdm import tqdm
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
        output_mode_3d: list[OutputMode3D],
        output_3d_num: int = 20
    ): 
        if output_mode_3d is None:
            output_mode_3d = [OutputMode3D.COR_LABEL, OutputMode3D.COR_MESH]
            
        self.image_dir = image_dir
        self.coronary_dir = coronary_dir
        self.cavity_dir = cavity_dir
        self.output_root = output_root
        
        self.dataset_name = dataset_name
        output_model_3d = set(OutputMode3D(m) for m in output_mode_3d)
        self.output_mode_3d = output_model_3d
        self.output_3d_num = output_3d_num
        self.save_buffer_size = 4
        self.save_worker_num = 4
        assert self.output_3d_num > 1, "output_3d_num should be greater than 1"

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

    def _gen_4D(
        self,
        reader: RBFReader,
        case_name: str,
        coronary_type: Literal["LCA", "RCA"]
    ):
        case_name = f"{self.dataset_name}__{case_name}"
        output_case_dir = self.output_root / case_name
        output_case_dir.mkdir(parents=True, exist_ok=True)
        worker_errors: list[Exception] = []
        error_lock = Lock()
        queue_done_token = object()

        def save_cor_label(output_path: Path, label, centering_affine):
            save_nii(output_path, label, centering_affine, is_label=True)

        def save_cor_mesh(output_path: Path, mesh_in_world):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mesh_in_world.save(str(output_path))

        def save_cavity_label(output_path: Path, cavity_label, affine):
            if output_path.exists():
                logging.warning(f"{output_path} already exists, skipping.")
            else:
                save_nii(output_path, cavity_label, affine, is_label=True)

        def save_volume(output_path: Path, volume, affine):
            if output_path.exists():
                logging.warning(f"{output_path} already exists, skipping.")
            else:
                save_nii(output_path, volume, affine)

        def save_phase_data(phase_str: str, data, affine):
            if OutputMode3D.COR_LABEL in self.output_mode_3d:
                output_path = output_case_dir / f"{case_name}_cor-label_{coronary_type}" / f"phase_{phase_str}.nii.gz"
                save_cor_label(output_path, data.coronary.label, data.coronary.centering_affine)
            if OutputMode3D.COR_MESH in self.output_mode_3d:
                output_path = output_case_dir / f"{case_name}_cor-mesh_{coronary_type}" / f"phase_{phase_str}.vtk"
                save_cor_mesh(output_path, data.coronary.mesh_in_world)
            if OutputMode3D.CAVITY_LABEL in self.output_mode_3d:
                output_path = output_case_dir / f"{case_name}_cavity-label" / f"phase_{phase_str}.nii.gz"
                save_cavity_label(output_path, data.cavity_label, affine)
            if OutputMode3D.VOLUME in self.output_mode_3d:
                output_path = output_case_dir / f"{case_name}_volume" / f"phase_{phase_str}.nii.gz"
                save_volume(output_path, data.volume, affine)

        save_buffer: Queue = Queue(maxsize=self.save_buffer_size)

        def save_worker():
            while True:
                item = save_buffer.get()
                try:
                    if item is queue_done_token:
                        return

                    phase_str, data, affine = item
                    save_phase_data(phase_str, data, affine)
                except Exception as error:
                    with error_lock:
                        worker_errors.append(error)
                    logging.exception(f"Save worker failed for {case_name} at phase {phase_str}: {error}")  #type: ignore
                finally:
                    save_buffer.task_done()
        
        workers = [Thread(target=save_worker, name=f"save-worker-{i}", daemon=False) for i in range(self.save_worker_num)]
        for worker in workers:
            worker.start()

        try:
            for phase in tqdm(np.linspace(0, 1, self.output_3d_num, endpoint=False), desc=f"Generating 4D data - {case_name} - {coronary_type}"):
                cardiac_phase = CardiacPhase(phase)
                phase_str = cardiac_phase.to_str(precision=2, has_decimal_point=False)
                data = reader.get_data(cardiac_phase, coronary_type)
                affine = reader.volume_affine
                save_buffer.put((phase_str, data, affine))

            # Wait until the entire case output has been saved.
            save_buffer.join()
        finally:
            for _ in workers:
                save_buffer.put(queue_done_token)
            for worker in workers:
                worker.join()

        if worker_errors:
            raise RuntimeError(f"{case_name} has {len(worker_errors)} save worker errors. Check processing_errors.log for details.")


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
                    self._gen_4D(reader, case_name, coronary_type)
                except Exception as e:
                    logging.exception(f"{image_nii} with {coronary_type} failed due to error: {e}")
                    continue

@app.default()
def main(
    dataset_names: list[Literal["asoca-normal", "asoca-diseased"]],
    output_root: Path,
    output_mode_3d: list[OutputMode3D] = [OutputMode3D.COR_LABEL, OutputMode3D.COR_MESH, OutputMode3D.CAVITY_LABEL, OutputMode3D.VOLUME],
    output_3d_num: int = 20
):
    """
    
    Parameters:
    ____
    dataset_names: list of dataset names to process, currently supports "asoca-normal" and "asoca-diseased"
    output_root: root directory to save the output data
    output_mode_3d: list of output modes for 3D data, can be "cor-label", "cor-mesh", "cavity-label" or "volume", default is ["cor-label", "cor-mesh"]
    output_3d_num: number of 3D phases to output in one phase, default is 20
    """
    for d in dataset_names:
        d = d.lower()
    
        match d:
            case "asoca-normal":
                print(f"{d} -- {output_root}")
                origin_dir = Path("/media/E/sj/Data/ASOCA/Normal/normal_gen_4d")
                dsa = BatchDVFToDSA(
                    image_dir=origin_dir/"CTCA_nii",
                    coronary_dir=origin_dir/"coronary",
                    cavity_dir=origin_dir/"cavity",
                    output_root=output_root,
                    dataset_name=d,
                    output_mode_3d=output_mode_3d,
                    output_3d_num=output_3d_num
                )
            case "asoca-diseased":
                print(f"{d} -- {output_root}")
                origin_dir = Path("/media/E/sj/Data/ASOCA/Diseased_partial")
                dsa = BatchDVFToDSA(
                    image_dir=origin_dir/"CTCA_nii",
                    coronary_dir=origin_dir/"coronary",
                    cavity_dir=origin_dir/"cavity",
                    output_root=output_root,
                    dataset_name=d,
                    output_mode_3d=output_mode_3d,
                    output_3d_num=output_3d_num
                )
            case _:
                print(f"{d} not supported")
                raise ValueError(f"{d} not supported")
    
        dsa.run()

if __name__ == "__main__":
    app()
