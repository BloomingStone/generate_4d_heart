from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field
from multiprocessing import get_context
import os

import torch
from torch import Tensor
import numpy as np
import pyvista as pv
import torchcpd
from tqdm import tqdm

from .data_reader import DataReader, DataReaderResult, Coronary, separate_coronary, load_nifti, get_coronary_centering_affine, get_mesh_in_world
from ... import NUM_TOTAL_PHASE
from ..cardiac_phase import CardiacPhase
from ..types import CoronaryType

@dataclass
class _Data:
    phase: CardiacPhase
    volume: Tensor
    cavity_label: Tensor
    affine: np.ndarray
    lca_label: Tensor
    rca_label: Tensor
    lca_mesh_in_world: pv.PolyData | None = None
    rca_mesh_in_world: pv.PolyData | None = None

def mapper(args: tuple[int, _Data, CoronaryType, np.ndarray, torch.device]) -> tuple[int, CoronaryType, pv.PolyData]:
    index, data, coronary_type, affine, device = args
    if coronary_type == CoronaryType.LCA:
        res = get_mesh_in_world(data.lca_label, affine, device, max_points=2000)
    else:
        res = get_mesh_in_world(data.rca_label, affine, device, max_points=2000)
    return (index, coronary_type, res)

class VolumesReader(DataReader):
    def __init__(
        self, 
        image_dir: Path,
        cavity_dir: Path,
        coronary_dir: Path,
        device: torch.device = torch.device("cuda:0")
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
        self.device = device
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
        
        self.data: list[_Data] = []
        for index, (img_file, cav_file, cor_file) in tqdm(
            enumerate(zip(image_file_list, cavity_file_list, coronary_file_list)),
            desc="Loading data", total=len(image_file_list)
        ):
            phase = CardiacPhase.from_index(index, self.n_phases)

            volume, affine = load_nifti(img_file)
            if volume.max() < 2.0:
                volume *= 2**16  # recover image uses float to represent 16 bit
                volume -= 1000
            
            cavity_label, _ = load_nifti(cav_file, is_label=True)
            coronary_label, _ = load_nifti(cor_file, is_label=True)

            lca_label, rca_label = separate_coronary(coronary_label, self.device)
            
            if index == 0:
                self._lca_centering_affine = get_coronary_centering_affine(lca_label, self._origin_volume_affine, self.device)
                self._rca_centering_affine = get_coronary_centering_affine(rca_label, self._origin_volume_affine, self.device)

            self.data.append(_Data(
                phase=phase,
                volume=volume.cpu(),
                cavity_label=cavity_label.cpu(),
                lca_label=lca_label.cpu(),
                rca_label=rca_label.cpu(),
                affine=affine,
            ))
        
        num_workers = torch.cuda.get_device_properties(self.device).total_memory / self.data[0].lca_label.numel() // 8
        num_cpu = os.process_cpu_count()
        assert num_cpu is not None
        num_workers = max(min(num_workers, num_cpu//2), 1)
        
        tasks: list[tuple[int, _Data, CoronaryType, np.ndarray, torch.device]] = []
        for index, data in enumerate(self.data):
            tasks.append((index, data, CoronaryType.LCA, self._lca_centering_affine, self.device))
            tasks.append((index, data, CoronaryType.RCA, self._rca_centering_affine, self.device))
        
        print(f"Using {num_workers} workers to generate mesh")
        with get_context("spawn").Pool(num_workers) as executor:
            results: list[tuple[int, CoronaryType, pv.PolyData]] = list(executor.map(mapper, tasks))
        
        # results = [mapper(task) for task in tasks]
        
        
        for index, type, res in results:
            if type == CoronaryType.LCA:
                self.data[index].lca_mesh_in_world = res
            else:
                self.data[index].rca_mesh_in_world = res
        
        # # use torchcpd to align mesh but failed
        # lca_template = None
        # rca_template = None
        # for data in tqdm(self.data, desc="Aligning mesh"):
        #     if lca_template is None or rca_template is None:
        #         lca_template = data.lca_mesh_in_world
        #         rca_template = data.rca_mesh_in_world
        #         continue
            
        #     new_lca_points, _ = torchcpd.DeformableRegistration(
        #         X=data.lca_mesh_in_world.points, 
        #         Y=lca_template.points, device=self.device
        #     ).register()
        #     new_rca_points, _ = torchcpd.DeformableRegistration(
        #         X=data.rca_mesh_in_world.points, 
        #         Y=rca_template.points, device=self.device
        #     ).register()
            
        #     new_lca_mesh = lca_template.copy()
        #     new_lca_mesh.points = new_lca_points.cpu().numpy()
        #     new_rca_mesh = rca_template.copy()
        #     new_rca_mesh.points = new_rca_points.cpu().numpy()
            
        #     data.lca_mesh_in_world = new_lca_mesh
        #     data.rca_mesh_in_world = new_rca_mesh
    

    def get_phase_0_data(self, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        return self._get_data_at_index(0, CoronaryType(coronary_type))

    def _get_data_at_index(self, index: int, coronary_type: CoronaryType) -> DataReaderResult:
        data = self.data[index]
        if coronary_type == CoronaryType.LCA:
            coronary_label = data.lca_label
            coronary_centering_affine = self._lca_centering_affine
            mesh = data.lca_mesh_in_world
        else:
            coronary_label = data.rca_label
            coronary_centering_affine = self._rca_centering_affine
            mesh = data.rca_mesh_in_world
        
        assert mesh is not None
        return DataReaderResult(
            phase=data.phase,
            volume=data.volume.cpu(),
            cavity_label=data.cavity_label.cpu().to(torch.uint8),
            affine=self._origin_volume_affine,
            coronary=Coronary(
                type=coronary_type,
                label=coronary_label.cpu().to(torch.bool),
                centering_affine=coronary_centering_affine,
                mesh_in_world=mesh
            )
        )

    def get_data(self, phase: CardiacPhase, coronary_type: CoronaryType | Literal["LCA", "RCA"]) -> DataReaderResult:
        """
        Returns interpolated data for the given phase in [0,1).
        If phase matches an existing frame index exactly, returns cached result.
        Otherwise performs fast linear interpolation between the two nearest frames.
        """
        coronary_type = CoronaryType(coronary_type)
        
        idx0 = phase.closest_index_floor(self.n_phases)
        idx1 = phase.closest_index_ceil(self.n_phases)
        
        w = float(phase)*self.n_phases - idx0  # interpolation weight [0,1)

        if w < 1e-6:  # exactly at frame idx0
            return self._get_data_at_index(idx0, coronary_type)

        d0 = self.data[idx0]
        d1 = self.data[idx1]

        # === intensity volume interpolation ===
        vol_interp = (1 - w) * d0.volume + w * d1.volume

        # === cavity label interpolation ===
        if w < 0.5:
            cav_interp = d0.cavity_label
        else:
            cav_interp = d1.cavity_label

        # === coronary label interpolation (LCA & RCA) ===
        if coronary_type == CoronaryType.LCA:
            coronary_label = ((1 - w) * d0.lca_label.float() + w * d1.lca_label.float()) > 0.5
            coronary_centering_affine = self._lca_centering_affine
            assert d0.lca_mesh_in_world is not None and d1.lca_mesh_in_world is not None
            new_mesh = d0.lca_mesh_in_world.copy() if w < 0.5 else d1.lca_mesh_in_world.copy()
            # new_mesh.points = (1 - w) * d0.lca_mesh_in_world.points + w * d1.lca_mesh_in_world.points  # need align first ini __init__
        else:
            coronary_label = ((1 - w) * d0.rca_label.float() + w * d1.rca_label.float()) > 0.5
            coronary_centering_affine = self._rca_centering_affine
            assert d0.rca_mesh_in_world is not None and d1.rca_mesh_in_world is not None
            new_mesh = d0.rca_mesh_in_world.copy() if w < 0.5 else d1.rca_mesh_in_world.copy()
            # new_mesh.points = (1 - w) * d0.rca_mesh_in_world.points + w * d1.rca_mesh_in_world.points
        
        return DataReaderResult(
            phase=phase,
            volume=vol_interp.cpu(),
            cavity_label=cav_interp.cpu().to(torch.uint8),
            affine=self._origin_volume_affine,
            coronary=Coronary(
                type=coronary_type,
                label=coronary_label.cpu().to(torch.bool),
                centering_affine=coronary_centering_affine,
                mesh_in_world=new_mesh
            )
        )
    
    @property
    def lca_centering_affine(self) -> np.ndarray:
        return self._lca_centering_affine
    
    @property
    def rca_centering_affine(self) -> np.ndarray:
        return self._rca_centering_affine

