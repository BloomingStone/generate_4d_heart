from pathlib import Path
import shutil
import warnings

from generate_4d_heart.rotate_dsa.contrast_simulator import StaticIodineContrast
from generate_4d_heart.rotate_dsa.data_reader import VolumeDVFReader, CoronaryBoundLVLinearEnhancer
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.rotate_dsa import RotateDSA

def test_coronary_bound_lv(
    output_root_dir: Path,
    test_data_root_dir: Path,
):
    warnings.warn("move_enhancer only used in VolumeDVFReader which is now can be replaced by RBFReader, so this test may be less meaningful and may be removed in the future.", DeprecationWarning)
    rotate_cfg = RotatedParameters()
    
    data_dir = test_data_root_dir / "volume_with_dvf"
    reader = VolumeDVFReader(
        volume_nii=data_dir / "image.nii.gz",
        cavity_nii=data_dir / "cavity.nii.gz",
        coronary_nii=data_dir / "coronary.nii.gz",
        dvf_dir=data_dir / "dvf",
        roi_json=data_dir / "Normal_01.json",
        contrast_simulator=StaticIodineContrast(),
        movement_enhancer=CoronaryBoundLVLinearEnhancer(enhance_coronary="LCA")
    )
    
    drr = TorchDRR(rotate_cfg=rotate_cfg)
    
    dsa = RotateDSA(
        reader=reader,
        drr=drr
    )
    
    output_dir = output_root_dir / "enhance" / "myo_bound_lv"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    dsa.run_and_save(
        output_dir=output_dir,
        coronary_type="LCA"
    )
    
    cta_output = output_dir / "cta"
    test_total_phases = 2
    for phase in range(test_total_phases):
        data = reader.get_data(CardiacPhase.from_index(phase, test_total_phases), "LCA")
        data.save(cta_output)