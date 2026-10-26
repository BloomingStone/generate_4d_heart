from pathlib import Path
import numpy as np
from generate_4d_heart.rotate_dsa.fix_angle_label_render import Projector
from generate_4d_heart.rotate_dsa.data_reader import RBFLabelReader
from generate_4d_heart.rotate_dsa.cardiac_phase import CardiacPhase
from generate_4d_heart.types import CoronaryType
from generate_4d_heart.ssm import SSMReader

if __name__ == "__main__":
    projector = Projector(
        reader=RBFLabelReader(
            cavity_nii=Path("test/test_data/volume_with_dvf/cavity.nii.gz"),
            coronary_nii=Path("test/test_data/volume_with_dvf/coronary.nii.gz"),
            ssm_reader=SSMReader(refine_steps=0),
        ),
    )
    
    alphas = np.linspace(0, 180, 90)
    # angles = [(0, 0), (30, 0), (60, 0), (90, 0), (120, 0), (150, 0), (180, 0)]
    angles = [(alpha, 0) for alpha in alphas]
    N = len(angles)
    phases = [CardiacPhase.from_time(i, N//3) for i in range(N)]
    projector.project_and_save(
        phase=phases,
        coronary_type=CoronaryType.LCA,
        angles_degree=angles,
        save_dir=Path("output/label_projections"),
        save_gif=True,
    )