from pathlib import Path

import torch

from .data_reader import DataReader, DataReaderResult, separate_coronary, load_nifti
from .. import NUM_TOTAL_PHASE, NUM_TOTAL_CAVITY_LABEL
from ..cardiac_phase import CardiacPhase


class VolumesReader(DataReader):
    def __init__(
        self, 
        image_dir: Path,
        cavity_dir: Path,
        coronary_dir: Path,
    ):
        """
        Volume data reader for 4D cardiac MRI.

        The data is assumed to be in the following format:
        - image_dir: contains 4D volume images
        | - phase_000.nii.gz   # do not use phase_0.nii which may cause ordering issues
        | - phase_001.nii.gz
        | - ...
        - cavity_dir: contains 4D cavity label images
        | - phase_000.nii.gz
        | - phase_001.nii.gz
        | - ...
        - coronary_dir: contains 4D coronary label images
        | - phase_000.nii.gz
        | - phase_001.nii.gz
        | - ...
        """
        image_file_list = sorted(image_dir.glob("*.nii*"))
        cavity_file_list = sorted(cavity_dir.glob("*.nii*"))
        coronary_file_list = sorted(coronary_dir.glob("*.nii*"))
        
        assert len(image_file_list) == len(cavity_file_list) == len(coronary_file_list), "Mismatch in number of files"
        
        self.n_phases = len(image_file_list)
        
        assert self.n_phases == NUM_TOTAL_PHASE, f"For now, we only support {NUM_TOTAL_PHASE} phases. But the data has {self.n_phases} phases."
        
        volume_0, affine_0 = load_nifti(image_file_list[0])
        self.origin_image_size = volume_0.shape[2:]   #type: ignore
        self.origin_image_affine = affine_0
        assert len(self.origin_image_size) == 3, f"Image size must be 3D, but got {self.origin_image_size}"
        
        self.data: list[DataReaderResult] = []
        for index, (img_file, cav_file, cor_file) in enumerate(zip(image_file_list, cavity_file_list, coronary_file_list)):
            phase = CardiacPhase.from_index(index, self.n_phases)

            volume, affine = load_nifti(img_file)
            cavity_label, _ = load_nifti(cav_file, is_label=True)
            coronary_label, _ = load_nifti(cor_file, is_label=True)

            lca_label, rca_label = separate_coronary(coronary_label)

            self.data.append(DataReaderResult(
                phase=phase,
                volume=volume.cpu(),
                cavity_label=cavity_label.cpu(),
                lca_label=lca_label.cpu(),
                rca_label=rca_label.cpu(),
                affine=affine
            ))
        
        
    
    def get_data(self, phase: CardiacPhase) -> DataReaderResult:
        """
        Returns interpolated data for the given phase in [0,1).
        If phase matches an existing frame index exactly, returns cached result.
        Otherwise performs fast linear interpolation between the two nearest frames.
        """
        idx0 = phase.closest_index_floor(self.n_phases)
        idx1 = phase.closest_index_ceil(self.n_phases)
        
        w = float(phase)*self.n_phases - idx0  # interpolation weight [0,1)

        if w < 1e-6:  # exactly at frame idx0
            return self.data[idx0]

        d0, d1 = self.data[idx0], self.data[idx1]

        # === intensity volume interpolation ===
        vol_interp = (1 - w) * d0.volume + w * d1.volume

        # === cavity label interpolation ===
        # use soft interpolation then rounding
        if w < 0.5:
            cav_interp = d0.cavity_label
        else:
            cav_interp = d1.cavity_label
        cav_interp = torch.clamp(cav_interp, 0, NUM_TOTAL_CAVITY_LABEL)

        # === coronary label interpolation (LCA & RCA) ===
        # binary -> linear then threshold (faster than morphological blending)
        lca_interp = ((1 - w) * d0.lca_label.float() + w * d1.lca_label.float()) > 0.5
        rca_interp = ((1 - w) * d0.rca_label.float() + w * d1.rca_label.float()) > 0.5

        return DataReaderResult(
            phase=phase,
            volume=vol_interp,
            cavity_label=cav_interp,
            lca_label=lca_interp,
            rca_label=rca_interp,
            affine=d0.affine  # assume same affine
        )

