import torch

from .contrast_simulator import ContrastSimulator


class ThresholdMultipliContrast(ContrastSimulator):
    def __init__(
        self, 
        lung_threshold: int = -600,
        heart_threshold: int = 0,
        bone_threshold: int = 600,
        lung_alpha: float = 1,
        heart_alpha: float = 0.3,
        bone_alpha: float = 1.0,
        coronary_alpha: float = 12.0,
        mu_water: float = 0.020,
    ):
        """
        adjust the contrast of coronary and cavity by simple multiplication
        """
        self.lung_threshold = lung_threshold
        self.heart_threshold = heart_threshold
        self.bone_threshold = bone_threshold
        self.lung_alpha = lung_alpha
        self.heart_alpha = heart_alpha
        self.bone_alpha = bone_alpha
        self.coronary_alpha = coronary_alpha
        self.mu_water = mu_water
    
    def preprocess(
        self,
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
    ) -> torch.Tensor:
        # Convert HU to baseline attenutation and apply tissue-specific scalings
        density = ori_volume.clone()
        density = density / 1000.0 * self.mu_water + self.mu_water

        air = torch.where((-1000 < ori_volume) & (ori_volume <= self.lung_threshold))
        lung = torch.where((self.lung_threshold < ori_volume) & (ori_volume <= self.heart_threshold))
        heart = torch.where((self.heart_threshold < ori_volume) & (ori_volume <= self.bone_threshold))
        bone = torch.where(ori_volume > self.bone_threshold)

        density[air] = density[lung].min() if len(lung[0]) > 0 else float(self.lung_threshold)
        density[lung] *= self.lung_alpha
        density[heart] *= self.heart_alpha
        density[bone] *= self.bone_alpha

        # ensure coronary baseline set to water-like baseline
        # caller will apply coronary-specific enhancement in `simulate`
        density[ori_volume < -2000] = 0
        return density
    
    def simulate(
        self, 
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor
    ) -> torch.Tensor:
        # Expect `ori_volume` to be preprocessed baseline attenuation map
        assert self.contrast_change_over_time == False, "ThresholdMultipliContrast does not support contrast change over time"
        density = ori_volume.clone()
        assert density.dtype == torch.float32 or density.dtype == torch.float16
        assert coronary_label.dtype == torch.bool

        # For simulate, scale coronary voxels from baseline
        coronary_idx = torch.where(coronary_label)
        if coronary_idx[0].numel() > 0:
            density[coronary_idx] = density[coronary_idx] * self.coronary_alpha
        return density


    
    def simulate_with_time(
        self, 
        ori_volume: torch.Tensor,
        cavity_label: torch.Tensor,
        coronary_label: torch.Tensor,
        time: float
    ) -> torch.Tensor:
        import warnings
        warnings.warn("ThresholdMultipliContrast does not support contrast change over time, `simulate_with_time` will ignore `time` input and return the same result as `simulate`")
        return self.simulate(ori_volume, cavity_label, coronary_label)