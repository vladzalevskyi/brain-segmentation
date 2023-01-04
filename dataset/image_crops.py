import numpy as np
import pandas as pd
from cv2 import BORDER_CONSTANT, copyMakeBorder
from skimage.util import montage
from skimage.util.shape import view_as_windows
import utils
from dataset.roi_extraction import slice_image, padd_image

def slice_image(image: np.ndarray, window_size: int, stride: int):
    """
    Slices image with a sliding window of given size and stride
    Args:
            image (np.ndarray): Image to be sliced.
            window_size (int): Window size. Should be divisible by stride.
            stride (int): Stride. Must be multiple of the window_size.
    Returns:
            np.ndarray: Numpy view of the sliced image patches
                        of shape (n_patches, w_size, w_size)
    """
    # TODO: This should contemplate the case in which the stride is larger than the patch size
    # if window_size % stride != 0:
    #     raise(ValueError("Window size must be a multiple of stride"))
    img_pathces = view_as_windows(image, window_size, stride)
    shape = img_pathces.shape
    return img_pathces.reshape((shape[0] * shape[1], shape[2], shape[3]))



class ImgCropsDataset():
    """Dataset of patches obtained from a single image"""
    def __init__(
        self,
        img: np.ndarray,
        patch_size: int = 224,
        stride: int = 100,
        min_breast_fraction_patch: float = None,
        normalization: str = 'z_score'
    ):
        """
        Args:
            img (np.ndarray): Image to process
            patch_size (int, optional): Defaults to 224.
            stride (int, optional): Defaults to 100.
            min_breast_fraction_patch (float, optional): Minimum of breast tissue that the patch
                should have in order to be classified. Defaults to None.
        """
        # instatiate atributes
        self.patch_size = patch_size
        self.stride = stride
        self.min_breast_frac = min_breast_fraction_patch
        self.normalization = normalization

        # extract patches equally from image and the mask
        img = padd_image(img, self.patch_size)
        self.image_patches = slice_image(img, window_size=self.patch_size, stride=self.stride)

        # calculate patches coordinates
        bbox_coordinates = []
        row_num, col_num, _, __ = view_as_windows(img, self.patch_size, self.stride).shape
        for col in range(row_num):
            row_idx = [((row * self.stride, col * self.stride),
                        (self.patch_size + row * self.stride,
                        self.patch_size + col * self.stride)) for row in range(col_num)]
            bbox_coordinates.extend(row_idx)
        self.bbox_coordinates = np.array(bbox_coordinates)

        if self.min_breast_frac is not None:
            breast_pixels = np.array([(roi != 0).sum() for roi in self.image_patches])
            breast_fraction = breast_pixels / (self.patch_size*self.patch_size)
            self.breast_fraction_selection = np.where(
                breast_fraction >= self.min_breast_frac, True, False)
            self.image_patches = self.image_patches[self.breast_fraction_selection, :, :]
            self.bbox_coordinates = self.bbox_coordinates[self.breast_fraction_selection, :, :]

    def __len__(self):
        return self.image_patches.shape[0]

    def __getitem__(self, idx):
        img = self.image_patches[idx, :, :]
        if img.any():
            if self.normalization == 'min_max':
                img = utils.min_max_norm(img, 1).astype('float32')
            elif self.normalization == 'z_score':
                img = utils.z_score_norm(img, non_zero_region=True)
        else:
            img = img.astype('float32')

        # to RGB
        # img = np.expand_dims(img, 0)
        # img = np.repeat(img, 3, axis=0)
        return {
            'img': img,
            'location': self.bbox_coordinates[idx, :, :],
        }