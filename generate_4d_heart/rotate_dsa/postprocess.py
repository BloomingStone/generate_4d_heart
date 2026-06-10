import torch
import torch.nn.functional as F
import numpy as np


def postprocess_drr(
        x: torch.Tensor,
        *,
        I0: float = 1.0,
        gamma: float = 0.1,     # enhance dark region which usually contains coronary, the value is determined empirically by visual inspection
        noise_std: float = 0.00,
        eps: float = 1e-8,
        output_uint8: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Post-process DRR image
    input -> exponentiate(line integral to intensity) -> normalize -> (gamma) -> (noise) -> (uint8) -> output
    Args:
        x: (T, H, W) DRR original line integral of density
        I0: float, initial intensity
        gamma: float, gamma correction
        noise_std: float, noise std for differentiable noise if noise_std > 0
        eps: float, used in normalization to avoid division by zero
        output_uint8: bool, if True, output is uint8, else float32 in [0,1]
    """
    I = x
    
    # -------------------------------
    # Gamma correction
    # -------------------------------
    if gamma != 1.0:
        I = torch.pow(I, gamma)
    
    # -------------------------------
    # Normalize [0,1]
    # -------------------------------
    # Use robust percentile-based normalization across the whole sequence (offline)
    def _percentile_value(t: torch.Tensor, q: float) -> torch.Tensor:
        # t: torch tensor (any shape), returns scalar tensor on same device
        try:
            # torch.quantile exists in recent versions
            return torch.quantile(t.flatten(), q / 100.0)
        except Exception:
            # fallback to numpy
            arr = t.cpu().numpy().ravel()
            val = np.percentile(arr, q)
            return torch.tensor(val, device=t.device, dtype=t.dtype)

    p_low = _percentile_value(I, 0.01)
    p_high = _percentile_value(I, 99.9)
    denom = (p_high - p_low).clamp_min(eps)
    I = (I - p_low) / denom
    I = I.clamp(0.0, 1.0)

    # -------------------------------
    # Noise
    # -------------------------------
    if noise_std > 0:
        I = I + torch.randn_like(I) * noise_std
        I = torch.clamp(I, 0, 1)
    
    if output_uint8:
        I = (I * 255).round().to(torch.uint8)
    
    post_process_meta = {
        "I0": I0,
        "gamma": gamma,
        "p_low": p_low.item(),
        "p_high": p_high.item(),
        "noise_std": noise_std,
    }

    return I, post_process_meta
