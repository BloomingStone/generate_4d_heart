from pathlib import Path
from random import random

from tqdm import tqdm
import torch

from generate_4d_heart.data_reader import VolumesReader, DataReader, VolumeDVFReader
from generate_4d_heart.cardiac_phase import CardiacPhase
from generate_4d_heart.saver import save_gif

def _read_and_save(reader: DataReader, output_dir: Path):
    # delete result in output_dir first
    for p in output_dir.iterdir():
        if p.is_dir():
            for f in p.iterdir():
                f.unlink()
            p.rmdir()
        else:
            p.unlink()
    
    for _ in tqdm(range(3), desc="Generating 3 random frames"):
        data = reader.get_data(CardiacPhase(random()))
        data.save(output_dir)
    
    F = 60
    fps = 30
    W, H, D = reader.origin_image_size
    frames_w = torch.rand(F, H, D)
    frames_h = torch.rand(F, W, D)
    frames_d = torch.rand(F, W, H)
    
    for phase_idx in tqdm(range(F), desc=f"Generating {F} continous frames in one cardiac cycle"):
        phase = CardiacPhase.from_index(phase_idx, F)
        
        data = reader.get_data(phase)
        volume = data.volume[0, 0]
        
        frames_w[phase_idx] = volume[W//2]
        frames_h[phase_idx] = volume[:, H//2]
        frames_d[phase_idx] = volume[:, :, D//2]
    
    def uniform(t: torch.Tensor) -> torch.Tensor:
        t = (t - t.min()) / (t.max() - t.min())
        return torch.clamp(t * 255, 0, 255).to(torch.uint8)
    
    frames_w = uniform(frames_w)
    frames_h = uniform(frames_h)
    frames_d = uniform(frames_d)
    
    save_gif(output_dir / "frames_w.gif", frames_w, fps)
    save_gif(output_dir / "frames_h.gif", frames_h, fps)
    save_gif(output_dir / "frames_d.gif", frames_d, fps)


def test_volumes_reader():
    test_dir = Path(__file__).parent / "test_data"
    volumes_dir = test_dir / "volumes"
    assert test_dir.exists()
    output_dir = volumes_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    reader = VolumesReader(
        image_dir=volumes_dir / "image",
        cavity_dir=volumes_dir / "cavity",
        coronary_dir=volumes_dir / "coronary"
    )
    
    _read_and_save(reader, output_dir)

def test_volume_dvf_reader():
    test_dir = Path(__file__).parent / "test_data"
    data_dir = test_dir / "volume_with_dvf"
    assert test_dir.exists()
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    reader = VolumeDVFReader(
        image_nii=data_dir / "image.nii.gz",
        cavity_nii=data_dir / "cavity.nii.gz",
        coronary_nii=data_dir / "coronary.nii.gz",
        dvf_dir=data_dir / "dvf",
        roi_json=data_dir / "Normal_01.json"
    )
    
    _read_and_save(reader, output_dir)