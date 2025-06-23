from scipy import ndimage
import numpy as np
from nibabel.nifti1 import Nifti1Image

from . import SSM_SHAPE


class ROI:
    """
    The cropped area and scaling ratio were calculated through the cardiac cavity label, with the aim of obtaining a cardiac cavity CTA of 144x144x128 for the subsequent 4D ddf generation
    """
    def __init__(
            self, 
            cavity: Nifti1Image, 
            padding: int = 10
    ):
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
        cropped_shape_z = min(mask.shape[2], size_z+padding*2)
        cropped_shape = (cropped_shape_xy, cropped_shape_xy, cropped_shape_z)
        x0 = max(0, x0 - padding)
        x1 = x0 + cropped_shape_xy
        y0 = max(0, y0 - padding)
        y1 = y0 + cropped_shape_xy
        z0 = max(0, z0 - padding)
        z1 = z0 + cropped_shape_z

        self.crop_box = (
            (x0, x1),
            (y0, y1),
            (z0, z1)
        )
        self.zoom_rate = np.array([
            SSM_SHAPE[i] / cropped_shape[i]
            for i in range(3)
        ])
        self.original_affine = cavity.affine
        assert self.original_affine is not None
        self.original_shape = cavity.shape

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
        T[:3, 3] = - np.array([x0, y0, z0])
        affine = affine @ np.linalg.inv(T)
        return Nifti1Image(image_data, affine)

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
        new_affine[:3, :3] = new_affine[:3, :3] @ np.diag(1/self.zoom_rate)
        return Nifti1Image(image_data, new_affine)

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
            background: the background image which has the same shape as original image, if not provided, the background will be zeros
        Returns:
            np.ndarray: the recovered image
        """
        if isinstance(cropped_and_zoomed_image, Nifti1Image):
            image_data = cropped_and_zoomed_image.get_fdata().copy()
        else:
            image_data = cropped_and_zoomed_image.copy()

        if is_label:
            image_data = ndimage.zoom(image_data, 1/self.zoom_rate, order=0).astype(np.uint8)
        else:
            image_data = ndimage.zoom(image_data, 1/self.zoom_rate, order=3)
        
        if background is not None:
            assert (
                background.shape == self.original_shape and
                background.affine is not None and
                self.original_affine is not None and 
                np.allclose(background.affine, self.original_affine, rtol=1e-5, atol=1e-8)
            )
            output_data = background.get_fdata().copy()
        else:
            output_data = np.zeros(self.original_shape, dtype=image_data.dtype)
        
        (x0, x1), (y0, y1), (z0, z1) = self.crop_box
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
                self.original_affine is not None and 
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
            "crop_box": self.crop_box,
            "zoom_rate": self.zoom_rate.tolist()
        }
