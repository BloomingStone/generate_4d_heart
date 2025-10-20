from unittest.mock import MagicMock
from pathlib import Path

import torch

from generate_4d_heart.rotate_dsa import RotateDSA
from generate_4d_heart.rotate_drr import TorchDRR, RotatedParameters
from generate_4d_heart import NUM_TOTAL_CAVITY_LABEL

def _rotate_dsa_mock(
    volume_shape: tuple[int, int, int],
    frame_num: int,
    output_dir: Path|None = None
):
    # ---- mock dependencies ----
    shape = (1, 1, *volume_shape)
    reader = MagicMock()
    reader.get_data.return_value = MagicMock(
        volume=torch.rand(shape),
        cavity_label=torch.randint(0, NUM_TOTAL_CAVITY_LABEL+1, shape),
        lca_label=torch.randint(0, 2, shape),
        rca_label=torch.randint(0, 2, shape),
        affine=torch.eye(4)
    )
    reader.origin_image_affine = torch.eye(4)
    reader.origin_image_size = volume_shape

    constrast_sim = MagicMock()
    constrast_sim.simulate.return_value = torch.rand(shape)
    drr = TorchDRR(rotate_cfg=RotatedParameters(total_frame=frame_num))
    dsa = RotateDSA(reader, constrast_sim, drr)
    
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        dsa.run_and_save(output_dir)
    else:
        frames, geo_json = dsa.run("LCA")
        assert isinstance(frames, torch.Tensor)
        assert frames.shape == (drr.rotate_cfg.total_frame, 1, *drr.image_size)
        assert "frames" in geo_json
        assert all("angle" in f for f in geo_json["frames"])

def test_rotate_dsa_mock_small():
    _rotate_dsa_mock((10, 10, 10), 120)

def test_rotate_dsa_mock_large():
    _rotate_dsa_mock((512, 512, 512), 3)

def test_rotate_dsa_mock_save(tmp_path):
    _rotate_dsa_mock((512, 512, 512), 3, tmp_path)

if __name__ == "__main__":
    test_rotate_dsa_mock_small()
