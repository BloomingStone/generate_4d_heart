from typing import Callable, TypeVar

import numpy as np
import torch

from . import SSM_DIRECTION

T = TypeVar('T', np.ndarray, torch.Tensor)

def get_maybe_flip_transform(affine: np.ndarray | None) -> Callable[[T], T]:
    if affine is None:
        return lambda x: x
    direction = np.diag(affine[:3, :3])
    axis = []
    for i in [-1, -2, -3]:
        if SSM_DIRECTION[i] * direction[i] < 0:
            axis.append(i)

    def flip(input: T) -> T:
        if isinstance(input, torch.Tensor):
            return torch.flip(input, dims=axis)
        elif isinstance(input, np.ndarray):
            return np.flip(input, axis=axis)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    return flip