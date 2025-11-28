import torch
import torch.nn.functional as F
import numpy as np


def differentiable_local_contrast_enhance(x, kernel_size=15, eps=1e-6):
    """
    similar to CLAHE in opencv2
    I_enhanced = (x - local_mean) / local_std
    """
    padding = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=x.device) / (kernel_size**2)

    # x: (T, H, W) → (T,1,H,W)
    x4 = x.unsqueeze(1)

    local_mean = F.conv2d(x4, weight, padding=padding)
    local_sqmean = F.conv2d(x4 * x4, weight, padding=padding)
    local_var = torch.clamp(local_sqmean - local_mean * local_mean, min=eps)
    local_std = torch.sqrt(local_var)

    out = (x4 - local_mean) / (local_std + eps)

    # 归一化到 [0,1]
    out = (out - out.min()) / (out.max() - out.min() + eps)
    return out.squeeze(1)


def differentiable_local_contrast_enhance_temporal(x: torch.Tensor, kernel_size: int = 15, kernel_t: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """
    Spatio-temporal local contrast enhancement using 3D convolution.
    Treat time as the depth dimension: x shape (T, H, W) -> conv3d over (T, H, W).
    Returns enhanced tensor with shape (T, H, W).
    This is an offline (non-causal, symmetric) filter by default (kernel_t centered).
    """
    assert x.ndim == 3, "Input must be (T, H, W)"

    T, H, W = x.shape
    pad_t = kernel_t // 2
    pad_hw = kernel_size // 2

    # separable uniform kernel in 3D: value = 1 / (kernel_t * kernel_size^2)
    kernel_val = 1.0 / (kernel_t * kernel_size * kernel_size)
    weight = torch.full((1, 1, kernel_t, kernel_size, kernel_size), fill_value=kernel_val, device=x.device, dtype=x.dtype)

    # (1,1,T,H,W)
    x5 = x.unsqueeze(0).unsqueeze(0)

    local_mean = F.conv3d(x5, weight, padding=(pad_t, pad_hw, pad_hw))
    local_sqmean = F.conv3d(x5 * x5, weight, padding=(pad_t, pad_hw, pad_hw))
    local_var = torch.clamp(local_sqmean - local_mean * local_mean, min=eps)
    local_std = torch.sqrt(local_var)

    out = (x5 - local_mean) / (local_std + eps)

    # normalize to [0,1] across the whole spatiotemporal block
    out = out - out.min()
    out = out / (out.max() + eps)

    return out.squeeze(0).squeeze(0)


def postprocess_drr_differentiable(
        x: torch.Tensor,
        mu: float = 1.0,
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

    p_low = _percentile_value(I, 0)
    p_high = _percentile_value(I, 99.5)
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
