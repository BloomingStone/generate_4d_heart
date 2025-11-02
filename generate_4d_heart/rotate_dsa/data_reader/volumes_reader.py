from pathlib import Path

import torch

from .data_reader import DataReader, DataReaderResult, separate_coronary, load_nifti, get_coronary_centering_affine
from ... import NUM_TOTAL_PHASE, NUM_TOTAL_CAVITY_LABEL
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
        self._origin_volume_size = volume_0.shape[2:]   #type: ignore
        self._origin_volume_affine = affine_0
        assert len(self._origin_volume_size) == 3, f"Image size must be 3D, but got {self._origin_volume_size}"
        
        self.data: list[DataReaderResult] = []
        for index, (img_file, cav_file, cor_file) in enumerate(zip(image_file_list, cavity_file_list, coronary_file_list)):
            phase = CardiacPhase.from_index(index, self.n_phases)

            volume, affine = load_nifti(img_file)
            if volume.max() < 2.0:
                volume *= 2**16  # recover image uses float to represent 16 bit
            
            sentinel_mask = (volume <= -2000)  # Some CTs may use -3023 or -2000 as 'sentinel' to mark invalid voxels
            min_value = volume[~sentinel_mask].min()
            volume[sentinel_mask] = min_value
            
            if volume.min() < -1000:
                volume += 1024   # add the offset of 1024

            cavity_label, _ = load_nifti(cav_file, is_label=True)
            coronary_label, _ = load_nifti(cor_file, is_label=True)

            lca_label, rca_label = separate_coronary(coronary_label)
            
            if index == 0:
                self.lca_centering_affine = get_coronary_centering_affine(lca_label, self._origin_volume_affine)
                self.rca_centering_affine = get_coronary_centering_affine(rca_label, self._origin_volume_affine)

            self.data.append(DataReaderResult(
                phase=phase,
                volume=volume.cpu(),
                cavity_label=cavity_label.cpu(),
                lca_label=lca_label.cpu(),
                rca_label=rca_label.cpu(),
                affine=affine,
                lca_centering_affine=self.lca_centering_affine,
                rca_centering_affine=self.rca_centering_affine
            ).to_device(torch.device("cpu")))
    
    
    def get_phase_0_data(self) -> DataReaderResult:
        return self.data[0]
    
    
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

        d0 = self.data[idx0].to_device(torch.device("cpu"))
        d1 = self.data[idx1].to_device(torch.device("cpu"))

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
            affine=d0.affine,
            lca_centering_affine=self.lca_centering_affine,
            rca_centering_affine=self.rca_centering_affine
        )

