from scipy import ndimage
import numpy as np
import cupy as cp
import torch
from nibabel.nifti1 import Nifti1Image
from pathlib import Path
from typing import TypeVar

from . import SSM_SHAPE

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
        mask = cavity_data > 0
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

        cropped_shape_xy = min(mask.shape[0], mask.shape[1], size_xy+padding*2)
        real_padding_xy = (cropped_shape_xy - size_xy) // 2
        cropped_shape_z = min(mask.shape[2], size_z+padding*2)
        real_padding_z = (cropped_shape_z - size_z) // 2
        cropped_shape = (cropped_shape_xy, cropped_shape_xy, cropped_shape_z)
        
        dx1 = mask.shape[0] - x1
        if dx1 > x0:
            x0 = max(0, x0 - real_padding_xy)
            x1 = x0 + cropped_shape_xy
        else:
            x1 = min(mask.shape[0], x1 + real_padding_xy)
            x0 = max(0, x1 - cropped_shape_xy)
            
        dy1 = mask.shape[1] - y1
        if dy1 > y0:
            y0 = max(0, y0 - real_padding_xy)
            y1 = y0 + cropped_shape_xy
        else:
            y1 = min(mask.shape[1], y1 + real_padding_xy)
            y0 = max(0, y1 - cropped_shape_xy)
        
        dz1 = mask.shape[2] - z1
        if dz1 > z0:
            z0 = max(0, z0 - real_padding_z)
            z1 = z0 + cropped_shape_z
        else:
            z1 = min(mask.shape[2], z1 + real_padding_z)
            z0 = max(0, z1 - cropped_shape_z)
        
        assert x0 >= 0 and x1 <= mask.shape[0] and y0 >= 0 and y1 <= mask.shape[1] and z0 >= 0 and z1 <= mask.shape[2]
        
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
    
    ImageData = TypeVar("image_data", np.ndarray, torch.Tensor, cp.ndarray)
    
    def crop_on_data(self, image: ImageData) -> ImageData:
        """ 
        Args:
            image_np: the 3d image data that needs to be cropped
        Returns:
            np.ndarray: the cropped image data
        """
        (x0, x1), (y0, y1), (z0, z1) = self.crop_box
        image_data = image[x0:x1, y0:y1, z0:z1]
        return image_data

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
        if is_label:
            image_data = ndimage.zoom(image_data, self.zoom_rate, order=0).astype(np.uint8)
        else:
            image_data = ndimage.zoom(image_data, self.zoom_rate, order=3)
        
        assert cropped_image.affine is not None
        new_affine = cropped_image.affine.copy()
        R = np.eye(4)
        R[:3, :3] = np.diag(1/self.zoom_rate)
        new_affine = new_affine @ R
        return Nifti1Image(image_data, new_affine)
    
    def crop_zoom_on_data(self, image: ImageData, is_label: bool) -> ImageData:
        """ 
        Args:
            image: the 3d image data that needs to be cropped and zoomed
            is_label: whether the image is a label. For label, the output will be binarized
        Returns:
            np.ndarray: the cropped and then zoomed image data
        """
        cropped_image = self.crop_on_data(image)
        image_data = cropped_image.get_fdata()
        if is_label:
            image_data = ndimage.zoom(image_data, self.zoom_rate, order=0).astype(np.uint8)
        else:
            image_data = ndimage.zoom(image_data, self.zoom_rate, order=3)
        
        return image_data

    def get_crop_box(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: the crop box, in the format of ((x0, x1), (y0, y1), (z0, z1))
        """
        return np.array(self.crop_box)
    
    def get_zoom_rate(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: the zoom rate, in the format of [x_rate, y_rate, z_rate], the rate = SSM_SHAPE / cropped_shape. and new_spacing = original_spacing / rate
        """
        return self.zoom_rate
    
    def recover(self, cropped_and_zoomed_image: Nifti1Image | np.ndarray, is_label: bool, background: Nifti1Image | None = None) -> Nifti1Image:
        """
        Recover the cropped and zoomed image to the original size based on the crop box and zoom rate.
        Args:
            cropped_and_zoomed_image: the cropped and zoomed image
            is_label: whether the image is a label. For label, the output will be binarized
            background: the background image which has the same shape as original image, if not provided, the background will be zeros
        Returns:
            Nifti1Image: the recovered image
        """
        if isinstance(cropped_and_zoomed_image, Nifti1Image):
            image_data = cropped_and_zoomed_image.get_fdata().copy()
        else:
            image_data = cropped_and_zoomed_image.copy()

        # Handle 4D case (3D + vector components)
        if image_data.ndim == 4:
            # Zoom each component separately
            zoomed_components = []
            for i in range(image_data.shape[-1]):
                component = image_data[..., i]
                if is_label:
                    zoomed = ndimage.zoom(component, 1/self.zoom_rate, order=0).astype(np.uint8)
                else:
                    zoomed = ndimage.zoom(component, 1/self.zoom_rate, order=3)
                zoomed_components.append(zoomed)
            image_data = np.stack(zoomed_components, axis=-1)
        else:
            # Standard 3D case
            if is_label:
                image_data = ndimage.zoom(image_data, 1/self.zoom_rate, order=0).astype(np.uint8)
            else:
                image_data = ndimage.zoom(image_data, 1/self.zoom_rate, order=3)
        
        # Handle output shape for vector fields (4D) vs scalar fields (3D)
        if image_data.ndim == 4:  # Vector field case
            output_shape = (*self.original_shape, image_data.shape[-1])
        else:  # Scalar field case
            output_shape = self.original_shape

        if background is not None:
            assert (
                background.shape == output_shape[:3] and  # Only check spatial dims
                background.affine is not None and
                np.allclose(background.affine, self.original_affine, rtol=1e-5, atol=1e-8)
            )
            output_data = background.get_fdata().copy()
            if output_data.ndim == 3 and image_data.ndim == 4:
                # Expand background to match vector field dims
                output_data = np.stack([output_data]*image_data.shape[-1], axis=-1)
        else:
            output_data = np.zeros(output_shape, dtype=image_data.dtype)
        
        (x0, x1), (y0, y1), (z0, z1) = self.crop_box
        
        if image_data.ndim == 4:  # Vector field
            for i in range(image_data.shape[-1]):
                output_data[x0:x1, y0:y1, z0:z1, i] = image_data[..., i]
        else:  # Scalar field
            output_data[x0:x1, y0:y1, z0:z1] = image_data

        return Nifti1Image(output_data, self.original_affine)
    
    def recover_cropped(self, cropped_image: Nifti1Image | np.ndarray, background: Nifti1Image | None = None) -> Nifti1Image:
        """
        Recover the cropped image to the original size based on the crop box and zoom rate.
        Args:
            cropped_image: the cropped image, shape = (W', H', D', ...) # the cropped image may have more dimensions which will be obtained to the output
            background: the background image which has the same shape as original image, if not provided, the background will be zeros
        Returns:
            np.ndarray: the recovered image
        """
        if isinstance(cropped_image, Nifti1Image):
            image_data = cropped_image.get_fdata().copy()
        else:
            image_data = cropped_image.copy()

        if len(image_data.shape) > 3 and background is None:
            output_shape = (*self.original_shape, *cropped_image.shape[3:])
        else:
            output_shape = self.original_shape

        if background is not None:
            assert (
                background.shape == self.original_shape and
                background.affine is not None and
                np.allclose(background.affine, self.original_affine, rtol=1e-5, atol=1e-8)
            )
            output_data = background.get_fdata().copy()
        else:
            output_data = np.zeros(output_shape, dtype=image_data.dtype)
        
        (x0, x1), (y0, y1), (z0, z1) = self.crop_box
        output_data[x0:x1, y0:y1, z0:z1] = image_data

        return Nifti1Image(output_data, self.original_affine)

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
