from pathlib import Path
import shutil

from generate_4d_heart.rotate_dsa.data_reader import StaticLabelReader
from generate_4d_heart.rotate_dsa.contrast_simulator import IdentityContrast
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters
from generate_4d_heart.rotate_dsa.rotate_dsa import RotateDSA


def test_only_static_label():
    test_root_dir = Path(__file__).parent
    test_data_root_dir = test_root_dir / "test_data"
    output_root_dir = test_root_dir / "output"
    
    data_dir = test_data_root_dir / "volume_with_dvf"
    reader = StaticLabelReader(
        cavity_path=data_dir / "cavity.nii.gz",
        coronary_path=data_dir / "coronary.nii.gz"
    )
    
    dsa = RotateDSA(
        reader=reader,
        constrast_sim=IdentityContrast(),
        drr=TorchDRR(
            rotate_cfg=RotatedParameters(
                total_frame=10,
                fps=10
            )
        )
    )
    
    output_dir = output_root_dir / "only_static_label"
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dsa.run_and_save(
        output_dir=output_dir,
        coronary_type="LCA",
        gray_reverse=False
    )