from pathlib import Path

import pytest
import torch


from generate_4d_heart.rotate_dsa.contrast_simulator import FlowContrast
from generate_4d_heart.rotate_dsa.data_reader.data_reader import load_nifti
from generate_4d_heart.saver import save_nii

from utils import simulator_factory, SimulatorName


test_angles = list(range(0, 180, 30))   # list of primary angles


@pytest.mark.parametrize("sim_name", tuple(SimulatorName))
def test_contrast_simulators(
    sim_name: SimulatorName,
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
    preprocessed_volume = simulator.preprocess(ori_volume, cavity_label, volume_affine)
    output_preprocessed_path = output_path / f"{sim_name}_preprocessed.nii.gz"
    save_nii(output_preprocessed_path, preprocessed_volume, affine=volume_affine)
    if simulator.contrast_change_over_time:
        simulated_volume = simulator.simulate_with_time(0.5, preprocessed_volume, cavity_label, coronary_label, volume_affine)
    else:
        simulated_volume = simulator.simulate(preprocessed_volume, cavity_label, coronary_label, volume_affine)
    
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

    import numpy as np

    early = simulator.simulate_with_time(0.05, ori_volume, cavity_label, coronary_label, np.eye(4))
    mid = simulator.simulate_with_time(0.2, ori_volume, cavity_label, coronary_label, np.eye(4))
    late = simulator.simulate_with_time(4.3, ori_volume, cavity_label, coronary_label, np.eye(4))

    # start point at near = (0, 0, 1, 4, 4) should enhance at t=0.1 and wash out at t=0.4
    near = (0, 0, 1, 4, 4)
    far = (0, 0, 5, 4, 4)

    assert early[near] < 0.2
    assert mid[near] > 0.5
    assert mid[far] < 0.2
    assert late[far] > 0.5


def test_cta_reset_contrast_preprocess_only_resets_inside_cavity():
    """
    CtaResetContrast.preprocess must reset CTA contrast ONLY inside the cavity
    label region:
      - enhanced blood-pool voxels inside the cavity -> reset to mu_water
      - myocardium voxels (inside cavity, low HU)    -> preserved
      - high-HU voxels outside the cavity (bone)     -> NOT reset
      - high-HU unlabeled aorta voxels               -> NOT reset
    """
    import numpy as np

    from generate_4d_heart.rotate_dsa.contrast_simulator import CtaResetContrast
    from generate_4d_heart import CavityLabel, MU_WATER

    D, H, W = 10, 10, 10
    hu = torch.zeros((D, H, W), dtype=torch.float32)
    cavity = torch.zeros((D, H, W), dtype=torch.uint8)

    # Left atrium (enhanced blood pool, label 4) with a spread of HU values
    la = (slice(2, 5), slice(2, 5), slice(2, 5))
    cavity[la] = CavityLabel.LA
    hu[la] = 400.0
    hu[la][0, 0, 0] = 300.0   # outlier low, still inside the LA block
    hu[la][2, 2, 2] = 500.0   # outlier high, still inside the LA block

    # Myocardium (label 1, low HU) inside the cavity
    myo = (slice(5, 8), slice(2, 5), slice(2, 5))
    cavity[myo] = CavityLabel.LV_MYO
    hu[myo] = 100.0

    # Bone (outside cavity, high HU)
    bone = (slice(0, 2), slice(0, 2), slice(0, 2))
    hu[bone] = 1200.0

    # Unlabeled aorta (outside cavity, high HU)
    aorta = (slice(8, 10), slice(8, 10), slice(8, 10))
    hu[aorta] = 800.0

    sim = CtaResetContrast()
    out = sim.preprocess(hu, cavity, np.eye(4))

    def mu_of(hu_val: float) -> float:
        return hu_val / 1000.0 * MU_WATER + MU_WATER

    # Enhanced blood pool inside the cavity -> reset to mu_water
    assert torch.isclose(out[3, 3, 3], torch.tensor(MU_WATER), atol=1e-6), \
        "Enhanced LA voxel inside cavity should be reset"
    # Myocardium preserved
    assert torch.allclose(out[myo], torch.full((3, 3, 3), mu_of(100.0)), atol=1e-6)
    # Bone preserved (NOT reset by a global threshold)
    assert torch.allclose(out[bone], torch.full((2, 2, 2), mu_of(1200.0)), atol=1e-6)
    # Unlabeled aorta preserved
    assert torch.allclose(out[aorta], torch.full((2, 2, 2), mu_of(800.0)), atol=1e-6)


def test_cta_reset_contrast_preprocess_empty_cavity_is_label_free():
    """
    With an all-zero cavity (as passed for the background CT in
    RBFReader._load_background), CtaResetContrast.preprocess must only apply
    the label-free HU->mu mapping and reset nothing.
    """
    import numpy as np

    from generate_4d_heart.rotate_dsa.contrast_simulator import CtaResetContrast
    from generate_4d_heart import MU_WATER

    hu = torch.tensor([
        [[[-1000.0], [0.0], [400.0], [1200.0]]],
    ], dtype=torch.float32)  # (1, 1, 4, 1)
    cavity = torch.zeros_like(hu, dtype=torch.uint8)

    sim = CtaResetContrast()
    out = sim.preprocess(hu, cavity, np.eye(4))

    expected = hu / 1000.0 * MU_WATER + MU_WATER
    assert torch.allclose(out, expected, atol=1e-6), \
        "Empty-cavity preprocessing must be plain HU->mu (no reset)"

# TODO need to be test
# air = torch.where(volume <= -800)
# soft_tissue = torch.where((-800 < volume) & (volume <= 350))
# bone = torch.where(350 < volume)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
