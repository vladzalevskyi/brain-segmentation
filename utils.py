import numpy as np
def z_score_norm(
    img: np.ndarray, mean: float = None, std: float = None, non_zero_region: bool = False
):
    if (mean is None):
        if non_zero_region:
            mean = img[img != 0].mean()
        else:
            mean = img.mean()
    if (std is None):
        if non_zero_region:
            std = img[img != 0].std()
        else:
            std = img.std()
    img = (img - mean) / std
    return img.astype('float32')


def min_max_norm(img: np.ndarray, max_val: int = None):
    """
    Scales images to be in range [0, 2**bits]
    Args:
        img (np.ndarray): Image to be scaled.
        max_val (int, optional): Value to scale images
            to after normalization. Defaults to None.
    Returns:
        np.ndarray: Scaled image with values from [0, max_val]
    """
    if max_val is None:
        max_val = np.iinfo(img.dtype).max
    img = (img - img.min()) / (img.max() - img.min()) * max_val
    return 