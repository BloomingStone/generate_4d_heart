from typing import Literal
import shutil

import pytest
import torch

from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters
from generate_4d_heart.saver import save_png
from generate_4d_heart.rotate_dsa.postprocess import postprocess_drr
from generate_4d_heart.rotate_dsa.contrast_simulator import FlowContrast

from utils import output_root_dir, get_reader, get_simulator

test_angles = list(range(0, 180, 30))   # list of primary angles

@pytest.mark.parametrize("reader_name", (
    "volumes_reader", "static_volume_reader"
))
@pytest.mark.parametrize("sim_name", (
    "multipli_contrast", "threshold_multipli_contrast"
))
@pytest.mark.parametrize("coronary_type", ["LCA", "RCA"])
def test_contrast_simulators(
    reader_name: str,
    sim_name: str,
    coronary_type: Literal["LCA", "RCA"],
):
    output_dir = output_root_dir / "contrast_sim" / f"{reader_name}_{sim_name}_{coronary_type}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    reader = get_reader(reader_name)
    data = reader.get_data(CardiacPhase(0.), coronary_type)
    coronary = data.coronary.label
    affine = data.coronary.centering_affine
    
    drr = TorchDRR(rotate_cfg=RotatedParameters(total_frame=1))
    simulator = get_simulator(sim_name)
    volume = simulator.simulate(
        ori_volume=data.coronary.volume,
        cavity_label=data.cavity_label,
        coronary_label=coronary
    )
    drr_res = drr.get_projections_at_degrees(
        angles=test_angles,
        volume=volume,
        coronary=coronary,
        affine=affine
    )
    drr_res = drr_res[:, 0:1]
    
    res = postprocess_drr(drr_res.squeeze())
    
    N, H, W = res.shape
    assert N == len(test_angles)
    
    for img, angle in zip(res, test_angles):
        save_png(
            output_path=output_dir / f"{angle}.png",
            image_2d=img
        )
        
    # data.save(output_dir)


def test_flow_contrast_time_delay_and_pulse_shape():
    ori_volume = torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32)
    cavity_label = torch.zeros((1, 1, 8, 8, 8), dtype=torch.uint8)
    coronary_label = torch.zeros((1, 1, 8, 8, 8), dtype=torch.bool)

    cavity_label[0, 0, 1, 4, 4] = 7  # aorta label seed near the inlet
    for x in range(1, 7):
        coronary_label[0, 0, x, 4, 4] = True

    simulator = FlowContrast(
        velocity=1.0,
        t_in=0.1,
        t_out=0.4,
        alpha=50.0,
    )

    early = simulator.simulate_with_time(ori_volume, cavity_label, coronary_label, time=0.05)
    mid = simulator.simulate_with_time(ori_volume, cavity_label, coronary_label, time=0.2)
    late = simulator.simulate_with_time(ori_volume, cavity_label, coronary_label, time=4.3)

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
    import sys
    pytest.main(["-s", __file__])
