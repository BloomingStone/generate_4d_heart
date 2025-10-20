from typing import Callable, TypeVar

import numpy as np
import torch

from .. import SSM_DIRECTION

T = TypeVar('T', np.ndarray, torch.Tensor)

class MaybeFlipTransform:
    """可调用的翻转变换类"""
    
    def __init__(self, affine: np.ndarray | None):
        self.affine = affine
        self.flip_axis = []
        self.flip_components = [False, False, False]
        self.need_flip = False
        
        if affine is not None:
            direction = np.diag(affine[:3, :3])
            for i in [-1, -2, -3]:
                if SSM_DIRECTION[i] * direction[i] < 0:
                    self.flip_axis.append(i)
                    self.flip_components[i] = True
            
            self.need_flip = any(self.flip_components)
    
    def __call__(self, input: T, is_vector_field: bool = False) -> T:
        """调用实例时执行翻转操作"""
        if not self.need_flip:
            return input
        
        if isinstance(input, torch.Tensor):
            res = torch.flip(input, dims=self.flip_axis)
        elif isinstance(input, np.ndarray):
            res = np.flip(input, axis=self.flip_axis)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        if is_vector_field:
            assert len(input.shape) >= 4 and input.shape[-4] == 3, "The shape of input vector field must be (..., 3, W, H, D)"
            for i in range(3):
                if self.flip_components[i]:
                    res[..., i, :, :, :] *= -1 #type: ignore
        return res