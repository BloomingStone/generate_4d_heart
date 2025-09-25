from typing import Callable, TypeVar

import numpy as np
import torch

from . import SSM_DIRECTION

T = TypeVar('T', np.ndarray, torch.Tensor)

def get_maybe_flip_transform(affine: np.ndarray | None) -> Callable[[T, bool], T]:
    if affine is None:
        return lambda x: x
    direction = np.diag(affine[:3, :3])
    flip_axis = []
    flip_components = [False, False, False]
    for i in [-1, -2, -3]:
        if SSM_DIRECTION[i] * direction[i] < 0:
            flip_axis.append(i)
            flip_components[i] = True
    
    need_flip = any(flip_components)

    def flip(input: T, is_vector_field: bool = False) -> T:
        if not need_flip:
            return input
        
        if isinstance(input, torch.Tensor):
            res = torch.flip(input, dims=flip_axis)
        elif isinstance(input, np.ndarray):
            res = np.flip(input, axis=flip_axis)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        if is_vector_field:
            assert len(input.shape) >= 4 and input.shape[-4] == 3, "The shape of input vector field must be (..., 3, W, H, D)"
            for i in range(3):
                if flip_components[i]:
                    res[..., i, :, :, :] *= -1
        return res

    return flip