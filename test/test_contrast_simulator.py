from typing import Literal
from pathlib import Path
import shutil

import pytest
import torch

from generate_4d_heart.rotate_dsa.contrast_simulator import MultipliContrast, ThresholdMultipliContrast, ContrastSimulator
from generate_4d_heart.rotate_dsa.data_reader import DataReader
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.rotate_dsa.rotate_drr import TorchDRR, RotatedParameters
from generate_4d_heart.saver import save_png

from utils import get_volumes_reader, get_volume_dvf_reader, output_root_dir

test_angles = list(range(0, 180, 30))

readers = {
    "volumes_reader": get_volumes_reader(),
    "volume_dvf_reader": get_volume_dvf_reader()
}

simulators = {
    "multipli_contrast": MultipliContrast(),
    "threshold_multipli_contrast": ThresholdMultipliContrast()
}

@pytest.mark.parametrize("reader_name, reader", readers.items())
@pytest.mark.parametrize("sim_name, simulator", simulators.items())
@pytest.mark.parametrize("coronary_type", ["LCA", "RCA"])
def test_contrast_simulators(
    reader_name: str,
    reader: DataReader,
    sim_name: str,
    simulator: ContrastSimulator,
    coronary_type: Literal["LCA", "RCA"],
):
    output_dir = output_root_dir / "contrast_sim" / f"{reader_name}_{sim_name}_{coronary_type}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    data = reader.get_data(CardiacPhase(0.))
    coronary = data.lca_label if coronary_type == "LCA" else data.rca_label
    
    drr = TorchDRR(rotate_cfg=RotatedParameters(total_frame=1))
    volume = simulator.simulate(
        ori_volume=data.volume,
        cavity_label=data.cavity_label,
        coronary_label=coronary
    )
    res = drr.get_projections_at_degrees(
        angles=test_angles,
        volume=volume,
        coronary=coronary,
        affine=data.affine
    )
    
    res = ((res - res.min()) / (res.max() - res.min())) * 255
    res = res.to(torch.uint8)
    res = torch.tensor(255) - res
    
    res = res.transpose(2, 3) # H, W -> W, H
    res = res.flip(-2)
    
    N, _, H, W = res.shape
    assert N == len(test_angles)
    
    for img, angle in zip(res, test_angles):
        save_png(
            output_path=output_dir / f"{angle}.png",
            image_2d=img
        )
        
    data.save(output_dir)

# TODO need to be test
# air = torch.where(volume <= -800)
# soft_tissue = torch.where((-800 < volume) & (volume <= 350))
# bone = torch.where(350 < volume)


if __name__ == "__main__":
    import sys
    pytest.main(["-s", __file__])
