from pathlib import Path

from generate_4d_heart.rotate_dsa.data_reader import VolumesReader, VolumeDVFReader, StaticVolumeReader
from generate_4d_heart.rotate_dsa.contrast_simulator import MultipliContrast, ThresholdMultipliContrast

test_root_dir = Path(__file__).parent
test_data_root_dir = test_root_dir / "test_data"
output_root_dir = test_root_dir / "output"

def get_volumes_reader():
    volumes_dir = test_data_root_dir / "volumes"
    assert test_data_root_dir.exists()
    
    return VolumesReader(
        image_dir=volumes_dir / "image",
        cavity_dir=volumes_dir / "cavity",
        coronary_dir=volumes_dir / "coronary"
    )

def get_volume_dvf_reader():
    data_dir = test_data_root_dir / "volume_with_dvf"
    return VolumeDVFReader(
        image_nii=data_dir / "image.nii.gz",
        cavity_nii=data_dir / "cavity.nii.gz",
        coronary_nii=data_dir / "coronary.nii.gz",
        dvf_dir=data_dir / "dvf",
        roi_json=data_dir / "Normal_01.json"
    )

def get_static_volume_reader():
    data_dir = test_data_root_dir / "volume_with_dvf"
    return StaticVolumeReader(
        image_path=data_dir / "image.nii.gz",
        cavity_path=data_dir / "cavity.nii.gz",
        coronary_path=data_dir / "coronary.nii.gz",
    )

readers = {
    "volumes_reader": get_volumes_reader(),
    "volume_dvf_reader": get_volume_dvf_reader(),
    "static_volume_reader": get_static_volume_reader()
}

simulators = {
    "multipli_contrast": MultipliContrast(),
    "threshold_multipli_contrast": ThresholdMultipliContrast()
}