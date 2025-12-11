import torch
import torch.nn.functional as F
import numpy as np


def postprocess_drr(
        x: torch.Tensor,
        *,
        mu: float = 0.1,    # 经验值（可能源于DRR对投影距离的处理）
        eps: float = 1e-8,
        gamma: float = 1.0,
        noise_std: float = 0.00,
        output_uint8: bool = True,
):
    """
    Post-process DRR image
    input -> exponentiate(line integral to intensity) -> normalize -> (gamma) -> (noise) -> (uint8) -> output
    Args:
        x: (T, H, W) DRR original line integral of density
        mu: float, attenuation coefficient
        gamma: float, gamma correction
        noise_std: float, noise std for differentiable noise if noise_std > 0
        eps: float, used in normalization to avoid division by zero
        output_uint8: bool, if True, output is uint8, else float32 in [0,1]
    """

    # -------------------------------
    # 1. Beer-Lambert: line integral → intensity
    # -------------------------------
    I = torch.exp( - mu * x)

    # -------------------------------
    # 2. Normalize [0,1]
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
    # 3. Gamma correction
    # -------------------------------
    if gamma != 1.0:
        I = torch.pow(I, gamma)

    # -------------------------------
    # 4. Noise
    # -------------------------------
    if noise_std > 0:
        I = I + torch.randn_like(I) * noise_std
        I = torch.clamp(I, 0, 1)
    
    if output_uint8:
        I = (I * 255).round().to(torch.uint8)

    return I
