from pathlib import Path

import pytest
import torch


from generate_4d_heart.rotate_dsa.contrast_simulator import FlowContrast
from generate_4d_heart.rotate_dsa.data_reader.data_reader import load_nifti
from generate_4d_heart.saver import save_nii

from utils import simulator_factory, SimulatorName


test_angles = list(range(0, 180, 30))   # list of primary angles


@pytest.mark.parametrize("sim_name", SimulatorName.to_tuple())
def test_contrast_simulators(
    sim_name: str,
    volume_dvf_dir: Path,
    output_root_dir: Path,
):
    ori_volume = volume_dvf_dir / "image.nii.gz"
    cavity_label = volume_dvf_dir / "cavity.nii.gz"
    coronary_label = volume_dvf_dir / "coronary.nii.gz"
    
    output_path = output_root_dir / "test_contrast_simulator"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ori_volume, volume_affine = load_nifti(ori_volume)
    cavity_label, _ = load_nifti(cavity_label, is_label=True)
    coronary_label, _ = load_nifti(coronary_label, is_label=True)

    simulator = simulator_factory(SimulatorName(sim_name))
    preprocessed_volume = simulator.preprocess(ori_volume, cavity_label)
    output_preprocessed_path = output_path / f"{sim_name}_preprocessed.nii.gz"
    save_nii(output_preprocessed_path, preprocessed_volume, affine=volume_affine)
    if simulator.contrast_change_over_time:
        simulated_volume = simulator.simulate_with_time(preprocessed_volume, cavity_label, coronary_label, time=0.5)
    else:
        simulated_volume = simulator.simulate(preprocessed_volume, cavity_label, coronary_label)
    
    output_simulated_path = output_path / f"{sim_name}_simulated.nii.gz"
    save_nii(output_simulated_path, simulated_volume, affine=volume_affine)


def test_flow_contrast_time_delay_and_pulse_shape():
    ori_volume = torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32)
    cavity_label = torch.zeros((1, 1, 8, 8, 8), dtype=torch.uint8)
    coronary_label = torch.zeros((1, 1, 8, 8, 8), dtype=torch.bool)

    cavity_label[0, 0, 0, 4, 4] = 7  # aorta label seed near the inlet
    for x in range(1, 7):
        coronary_label[0, 0, x, 4, 4] = True

    simulator = FlowContrast(
        velocity=1.0,
        t_in=0.1,
        t_out=0.4,
        alpha=50.0,
        mu_idodine=1.0,
        mu_water_dsa=0.0
    )

    early = simulator.simulate_with_time(ori_volume, cavity_label, coronary_label, time=0.05)
    mid = simulator.simulate_with_time(ori_volume, cavity_label, coronary_label, time=0.2)
    late = simulator.simulate_with_time(ori_volume, cavity_label, coronary_label, time=4.3)

    # start point at near = (0, 0, 1, 4, 4) should enhance at t=0.1 and wash out at t=0.4
    near = (0, 0, 1, 4, 4)
    far = (0, 0, 5, 4, 4)

    assert early[near] < 0.2
    assert mid[near] > 0.5
    assert mid[far] < 0.2
    assert late[far] > 0.5

# TODO need to be test
# air = torch.where(volume <= -800)
# soft_tissue = torch.where((-800 < volume) & (volume <= 350))
# bone = torch.where(350 < volume)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
