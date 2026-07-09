from pathlib import Path
from generate_4d_heart.rotate_dsa.rotate_drr.torch_drr import TorchDRR
from generate_4d_heart.rotate_dsa.rotate_dsa import RotateDSA
from utils import reader_factory, ReaderName, SimulatorName, simulator_factory

def test_projection(
    single_data_drr: TorchDRR,
    output_root_dir: Path
):
    _, reader=reader_factory(
        simulator_name=SimulatorName.STATIC_IODINE_CONTRAST,
        reader_name=ReaderName.STATIC_VOLUME_READER
    )
    dsa = RotateDSA(reader=reader, drr=single_data_drr, multi_threads=False)
    
    output_dir = output_root_dir / "test_projection"
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dsa.run_and_save(
        output_dir=output_dir,
        coronary_type="LCA",
        gif_fps=10
    )