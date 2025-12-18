from scipy import ndimage
import numpy as np
import torch
from nibabel.nifti1 import Nifti1Image
from pathlib import Path
from typing import overload

from . import SSM_SHAPE, ALL_CAVITY_LABEL

def load_array(
    array: tuple | list | np.ndarray | None, 
    default_array: np.ndarray, 
    target_shape: tuple[int, ...]
) -> np.ndarray:
    if array is None:
        res = default_array
    elif isinstance(array, np.ndarray):
        res = array
    elif isinstance(array, list) or isinstance(array, tuple):
        res = np.array(array)
    else:
        raise ValueError("The array must be a tuple, list or numpy.ndarray")
    
    if res.shape != target_shape:
        raise ValueError("The array must be a 144x144x128 numpy.ndarray")
    
    return res

# TODO 考虑增加对ScalarImage的支持
class ROI:
    """
    The cropped area and scaling ratio were calculated through the cardiac cavity label, with the aim of obtaining a cardiac cavity CTA of 144x144x128 for the subsequent 4D ddf generation
    """
    def __init__(
            self,
            cropped_box: tuple | list | np.ndarray | None = None,
            zoom_rate: tuple | list | np.ndarray | None = None,
            original_affine: tuple | list | np.ndarray | None = None,
            original_shape: tuple[int, int, int] | None = None
        ):
        self.crop_box = load_array(cropped_box, np.array(((0, 144), (0, 144), (0, 128))), (3, 2))
        self.zoom_rate = load_array(zoom_rate, np.array([1., 1., 1.]), (3,))
        self.original_affine = load_array(original_affine, np.eye(4), (4, 4))
        self.original_shape = original_shape if original_shape is not None else SSM_SHAPE
    
    @staticmethod
    def get_from_cavity( 
            cavity: Nifti1Image, 
            padding: int = 10
    ) -> "ROI":
        """
        Args:
            cavity: Path to the cavity label
            padding: the padding size around the cavity, the padding will be added to the both sides of the cavity as passible.
        """
        cavity_data = cavity.get_fdata()
        mask = np.zeros_like(cavity_data, dtype=bool)
        for label in ALL_CAVITY_LABEL:
            mask[cavity_data == label] = True
        if not mask.any():
            raise ValueError("The cavity label is empty")
        
        coords = np.where(mask)
        x0, x1 = coords[0].min(), coords[0].max()
        y0, y1 = coords[1].min(), coords[1].max()
        z0, z1 = coords[2].min(), coords[2].max()

        size_z = z1 - z0
        size_xy = max(x1 - x0, y1 - y0)
        x_center = (x0 + x1) // 2
        y_center = (y0 + y1) // 2
        x0 = int(x_center - size_xy // 2)
        x1 = x0 + size_xy
        y0 = int(y_center - size_xy // 2)
        y1 = y0 + size_xy

        cropped_shape_xy = min(mask.shape[0]-1, mask.shape[1]-1, size_xy+padding*2)
        real_padding_xy = (cropped_shape_xy - size_xy) // 2
        cropped_shape_z = min(mask.shape[2]-1, size_z+padding*2)
        real_padding_z = (cropped_shape_z - size_z) // 2
        
        dx1 = mask.shape[0] - 1 - x1
        if dx1 > x0:
            x0 = max(0, x0 - real_padding_xy)
            x1 = x0 + cropped_shape_xy
        else:
            x1 = min(mask.shape[0]-1, x1 + real_padding_xy)
            x0 = max(0, x1 - cropped_shape_xy)
            
        dy1 = mask.shape[1] - 1 - y1
        if dy1 > y0:
            y0 = max(0, y0 - real_padding_xy)
            y1 = y0 + cropped_shape_xy
        else:
            y1 = min(mask.shape[1]-1, y1 + real_padding_xy)
            y0 = max(0, y1 - cropped_shape_xy)
        
        dz1 = mask.shape[2] - 1 - z1
        if dz1 > z0:
            z0 = max(0, z0 - real_padding_z)
            z1 = z0 + cropped_shape_z
        else:
            z1 = min(mask.shape[2]-1, z1 + real_padding_z)
            z0 = max(0, z1 - cropped_shape_z)
        
        x0 = np.clip(x0, 0, mask.shape[0] - 1)
        x1 = np.clip(x1, 0, mask.shape[0] - 1)
        y0 = np.clip(y0, 0, mask.shape[1] - 1)
        y1 = np.clip(y1, 0, mask.shape[1] - 1)
        z0 = np.clip(z0, 0, mask.shape[2] - 1)
        z1 = np.clip(z1, 0, mask.shape[2] - 1)
        
        cropped_shape = (
            (x1 - x0), (y1 - y0), (z1 - z0)
        )
        
        roi = ROI()
        roi.crop_box = np.array([
            [x0, x1],
            [y0, y1],
            [z0, z1]
        ])
        roi.zoom_rate = np.array([
            SSM_SHAPE[i] / cropped_shape[i]
            for i in range(3)
        ])
        roi.original_affine = cavity.affine if cavity.affine is not None else np.eye(4)
        roi.original_shape = cavity.shape
        return roi

    def crop(self, image: Nifti1Image) -> Nifti1Image:
        """
        Args:
            image: the 3d image that needs to be cropped
        Returns:
            np.ndarray: the cropped image
        """
        (x0, x1), (y0, y1), (z0, z1) = self.crop_box
        image_data = image.get_fdata().copy()
        image_data = image_data[x0:x1, y0:y1, z0:z1]
        affine = image.affine
        assert affine is not None
        affine = affine.copy()
        T = np.eye(4)
        T[:3, 3] = np.array([x0, y0, z0])
        affine = affine @ T
        return Nifti1Image(image_data, affine)
    
    def get_affine_after_crop(self, affine: np.ndarray) -> np.ndarray:
        (x0, x1), (y0, y1), (z0, z1) = self.crop_box
        res = affine.copy()
        T = np.eye(4)
        T[:3, 3] = np.array([x0, y0, z0])
        res = res @ T
        return res
    
    def get_affine_after_crop_and_zoom(self, affine: np.ndarray) -> np.ndarray:
        raise NotImplementedError() # need to clearify the order of R and T
    
    @overload
    def crop_on_data(self, image: np.ndarray) -> np.ndarray: ...
    
    @overload
    def crop_on_data(self, image: torch.Tensor) -> torch.Tensor: ...
    
    def crop_on_data(self, image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """ 
        Args:
            image_np: the 3d image data that needs to be cropped
        Returns:
            np.ndarray: the cropped image data
        """
        (x0, x1), (y0, y1), (z0, z1) = self.crop_box
        return image[..., x0:x1, y0:y1, z0:z1]

    def _zoom_np(self, np_data: np.ndarray, is_label: bool) -> np.ndarray:
        if is_label:
            res = ndimage.zoom(np_data, self.zoom_rate, order=0)
            assert isinstance(res, np.ndarray)
            res = res.astype(np.uint8)
        else:
            res = ndimage.zoom(np_data, self.zoom_rate, order=3)
            assert isinstance(res, np.ndarray)
            res = res.astype(np.float32)
        return res
        
    
    def crop_zoom(self, image: Nifti1Image, is_label: bool) -> Nifti1Image:
        """ 
        Args:
            image: the 3d image that needs to be cropped and zoomed
            is_label: whether the image is a label. For label, the output will be binarized
        Returns:
            np.ndarray: the cropped and then zoomed image
        """
        cropped_image = self.crop(image)
        image_data = cropped_image.get_fdata()
        image_data = self._zoom_np(image_data, is_label)
        
        assert cropped_image.affine is not None
        new_affine = cropped_image.affine.copy()
        R = np.eye(4)
        R[:3, :3] = np.diag(1/self.zoom_rate)
        new_affine = new_affine @ R
        return Nifti1Image(image_data, new_affine)
    
    @overload
    def crop_zoom_on_data(self, image: np.ndarray, is_label: bool) -> np.ndarray: ...
    
    @overload
    def crop_zoom_on_data(self, image: torch.Tensor, is_label: bool) -> torch.Tensor: ...
    
    def crop_zoom_on_data(self, image: np.ndarray | torch.Tensor, is_label: bool) -> np.ndarray | torch.Tensor:
        """ 
        Args:
            image: the 3d image data that needs to be cropped and zoomed
            is_label: whether the image is a label. For label, the output will be binarized
        Returns:
            np.ndarray: the cropped and then zoomed image data
        """
        res = self.crop_on_data(image)
        if isinstance(res, np.ndarray):
            res = self._zoom_np(res, is_label)
        elif isinstance(res, torch.Tensor):
            if is_label:
                res = torch.nn.functional.interpolate(res, scale_factor=self.zoom_rate, mode='nearest')
            else:
                res = torch.nn.functional.interpolate(res, scale_factor=self.zoom_rate, mode='trilinear')
            assert isinstance(res, torch.Tensor)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        return res

    def get_crop_box(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: the crop box, in the format of ((x0, x1), (y0, y1), (z0, z1))
        """
        return np.array(self.crop_box)
    
    def get_roi_size_before_crop(self) -> tuple[int, int, int]:
        (x0, x1), (y0, y1), (z0, z1) = self.crop_box
        return (
            int(x1 - x0), int(y1 - y0), int(z1 - z0)
        )
    
    def get_zoom_rate(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: the zoom rate, in the format of [x_rate, y_rate, z_rate], the rate = SSM_SHAPE / cropped_shape. and new_spacing = original_spacing / rate
        """
        return self.zoom_rate

    def recover_cropped_tensor(self, cropped_image: torch.Tensor, background: torch.Tensor | None = None) -> torch.Tensor:
        """
        Recover the cropped image to the original size based on the crop box and zoom rate.
        Args:
            cropped_image: the cropped image, shape = (..., W', H', D')
            background: the background image which has the same shape as original image, if not provided, the background will be zeros
        Returns:
            torch.Tensor: the recovered image
        """
        if cropped_image.dim() > 3:
            output_shape = (*cropped_image.shape[:-3], *self.original_shape)
        else:
            output_shape = self.original_shape
            
        if background is not None:
            assert background.shape == output_shape
            output_data = background.clone()
        else:
            output_data = torch.zeros(output_shape, dtype=cropped_image.dtype)
        
        (x0, x1), (y0, y1), (z0, z1) = self.crop_box
        output_data[..., x0:x1, y0:y1, z0:z1] = cropped_image
        return output_data

    def to_dict(self) -> dict:
        """
        Returns:
            dict: the dict format of the ROI
        """
        return {
            "crop_box": self.crop_box.tolist(),
            "zoom_rate": self.zoom_rate.tolist(),
            "original_shape": self.original_shape,
            "original_affine": self.original_affine.tolist()
        }
    
    @staticmethod
    def from_dict(roi_dict: dict) -> "ROI":
        """
        Args:
            roi_dict: the dict format of the ROI
        Returns:
            ROI: the ROI object
        """
        return ROI(
            roi_dict["crop_box"],
            roi_dict["zoom_rate"],
            roi_dict["original_affine"],
            roi_dict["original_shape"]
        )
    
    @staticmethod
    def from_json(json_path: Path) -> "ROI":
        """
        Args:
            json_path: the path to the json file
        Returns:
            ROI: the ROI object
        """
        import json
        with open(json_path, "r") as f:
            roi_dict = json.load(f)
        return ROI.from_dict(roi_dict)


if __name__ == "__main__":
    ROI()
