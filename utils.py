
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

def resample2target(moving, target):
    """Resamples moving image to target image"""

    return sitk.Resample(moving, target.GetSize(),
                                    sitk.Transform(), 
                                    sitk.sitkLinear,
                                    target.GetOrigin(),
                                    target.GetSpacing(),
                                    target.GetDirection(),
                                    0,
                                    target.GetPixelID())


def print_img_info(selected_image, title='Train image:'):
    print(title)
    print('origin: ' + str(selected_image.GetOrigin()))
    print('size: ' + str(selected_image.GetSize()))
    print('spacing: ' + str(selected_image.GetSpacing()))
    print('direction: ' + str(selected_image.GetDirection()))
    print('pixel type: ' + str(selected_image.GetPixelIDTypeAsString()))
    print('number of pixel components: ' + str(selected_image.GetNumberOfComponentsPerPixel()))



# a simple function to plot an image
def plot1(fixed, title='', slice=128, figsize=(12, 12)):
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.imshow(sitk.GetArrayFromImage(fixed)[slice, :, :], cmap='gray', origin='lower')
    axs.set_title(title, fontdict={'size':26})
    axs.axis('off')
    plt.tight_layout()
    plt.show()
    
# a simple function to plot 3 images at once
def plot3(fixed, moving, transformed, labels=['Fixed', 'Moving', 'Moving Transformed'], slice=128):
    fig, axs = plt.subplots(1, 3, figsize=(24, 12))
    axs[0].imshow(sitk.GetArrayFromImage(fixed)[slice, :, :], cmap='gray', origin='lower')
    axs[0].set_title(labels[0], fontdict={'size':26})
    axs[0].axis('off')
    axs[1].imshow(sitk.GetArrayFromImage(moving)[slice, :, :], cmap='gray', origin='lower')
    axs[1].axis('off')
    axs[1].set_title(labels[1], fontdict={'size':26})
    axs[2].imshow(sitk.GetArrayFromImage(transformed)[slice, :, :], cmap='gray', origin='lower')
    axs[2].axis('off')
    axs[2].set_title(labels[2], fontdict={'size':26})
    plt.tight_layout()
    plt.show()

def convert_nda_to_itk(nda: np.ndarray, itk_image: sitk.Image):
    """From a numpy array, get an itk image object, copying information
    from an existing one. It switches the z-axis from last to first position.

    Args:
        nda (np.ndarray): 3D image array
        itk_image (sitk.Image): Image object to copy info from

    Returns:
        new_itk_image (sitk.Image): New Image object
    """
    new_itk_image = sitk.GetImageFromArray(np.moveaxis(nda, -1, 0))
    new_itk_image.SetOrigin(itk_image.GetOrigin())
    new_itk_image.SetSpacing(itk_image.GetSpacing())
    new_itk_image.CopyInformation(itk_image)
    return new_itk_image

def convert_itk_to_nda(itk_image: sitk.Image):
    """From an itk Image object, get a numpy array. It moves the first z-axis
    to the last position (np.ndarray convention).

    Args:
        itk_image (sitk.Image): Image object to convert

    Returns:
        result (np.ndarray): Converted nda image
    """
    return np.moveaxis(sitk.GetArrayFromImage(itk_image), 0, -1)
    
def padd_images_to_max(img1, img2):
    """Padds images so that they have the same size.
    
    Zero paddig is used on the right and bottom side of the image
    so that image with smallest dimensions is padded to the size of the largest one."""
    padding_filter = sitk.ConstantPadImageFilter()

    for dim in range(len(img1.GetSize())):
        padding_array = [0, 0, 0]
        padding_size = img1.GetSize()[dim] - img2.GetSize()[dim]
        padding_array[dim] = abs(padding_size)
        
        if img1.GetSize()[dim] > img2.GetSize()[dim]:
            img2 = padding_filter.Execute(img2, (0, 0, 0), padding_array, 0)
        else:
            img1 = padding_filter.Execute(img1, (0, 0, 0), padding_array, 0)
    return img1, img2

def register_image(fixed_image, moving_image, transformParameterMap=None, interpolator='nn'):
    """Perform registration of the moving image to fixed image. 
    
    Either learn the registration and return the result and transformParameterMap
    or use the one passed as an argument

    Args:
        fixed_image (sitk.Image): Fixed image
        moving_image (sitk.Image): Moving image
        transformParameterMap (SimpleITK.SimpleITK.ParameterMap, optional): Parameter map used for registration.Defaults to None.
            If None, the registration is learned and the learned parameter map is returned.
            Otherwise, uses the map passed to perform the transformation.
        interpolator (str, optional): If 'nn' changes the interpolator in the transformParameterMap
            to FinalNearestNeighborInterpolator. Otherwise, uses the one from the map. 
            Defaults to 'nn'.

    Returns:
        (transformed image (sitk.Image), transformParameterMap)
    """
    fixed_image, moving_image = padd_images_to_max(fixed_image, moving_image)
    if transformParameterMap is None:
        elastixImageFilter_non_rigid = sitk.ElastixImageFilter()
        elastixImageFilter_non_rigid.SetFixedImage(fixed_image)
        elastixImageFilter_non_rigid.SetMovingImage(moving_image)

        elastixImageFilter_non_rigid.Execute()
        transformParameterMap = elastixImageFilter_non_rigid.GetTransformParameterMap()
        return elastixImageFilter_non_rigid.GetResultImage(), transformParameterMap
        
    else:
        transformixImageFilter = sitk.TransformixImageFilter()
        # change interpolator to NN for label
        if interpolator == 'nn':
            for transfrom in transformParameterMap:
                transfrom['ResampleInterpolator'] = ('FinalNearestNeighborInterpolator',)

        transformixImageFilter.SetTransformParameterMap(transformParameterMap)
        transformixImageFilter.SetMovingImage(moving_image)
        transformixImageFilter.Execute()
        return transformixImageFilter.GetResultImage(), transformParameterMap
        
def convert_nda_to_itk(nda: np.ndarray, itk_image: sitk.Image):
    """From a numpy array, get an itk image object, copying information
    from an existing one. It switches the z-axis from last to first position.

    Args:
        nda (np.ndarray): 3D image array
        itk_image (sitk.Image): Image object to copy info from

    Returns:
        new_itk_image (sitk.Image): New Image object
    """
    new_itk_image = sitk.GetImageFromArray(np.moveaxis(nda, -1, 0))
    new_itk_image.SetOrigin(itk_image.GetOrigin())
    new_itk_image.SetSpacing(itk_image.GetSpacing())
    new_itk_image.CopyInformation(itk_image)
    return new_itk_image

def remap_labels(img: np.ndarray, mapping_dict = None):
    """Remap SynthSeg labels to GM, WM, CSF

    Args:
        img (np.ndarray): SynthSeg labes image
        mapping_dict (_type_, optional): Mapping dict. Defaults to None.

    Returns:
        np.ndarray: Segmentation image with new labels
    """
    if mapping_dict is None:
        mapping_dict = {0: [0], #background
                        1: [24, 4, 43, 14, 15, 44, 5], #CSF
                        2: [3, 8, 42, 47, 10, 49, 11, 50, 13, 52, 12, 51, 17, 53, 18, 54, 26, 58, 28, 60], #GM
                        3: [2, 16, 7, 41, 46] #WM
                        }

    ss_img_new = np.zeros(img.shape, dtype=np.int16)
    for key, value in mapping_dict.items():
        for v in value:
            ss_img_new[img == v] = int(key)

    return ss_img_new

def convert_itk_to_nda(itk_image: sitk.Image):
    """From an itk Image object, get a numpy array. It moves the first z-axis
    to the last position (np.ndarray convention).

    Args:
        itk_image (sitk.Image): Image object to convert

    Returns:
        result (np.ndarray): Converted nda image
    """
    return np.moveaxis(sitk.GetArrayFromImage(itk_image), 0, -1)

def tissue_model_segmentation(image, brain_mask, tissue_model):
    """
    Compute segmentation from a brain volume by mapping each gray level
    to a LUT of tissue models.

    Args:
        image (np.ndarray): 3D volume of brain
        brain_mask (np.ndarray): 3D volume (same size as image) with brain tissues GT != 0
        tissue_model (np.ndarray): LUT of tissue models, columns = [0 - CSF, 1 - WM, 2 - GM]

    Returns:
        result_discr (np.ndarray): Discrete segemented brain with labels 1 - CSF, 2 - WM, 3 - GM
        result_prb (np.ndarray): Tissue probabilities volumes. Last axis of array corresponds to
                                probability volumes: 0 - CSF, 1 - WM, 2 - GM
    """
    image_shape = image.shape
    result_discr = np.zeros((image_shape))
    result_prb = np.zeros((image_shape[0], image_shape[1], image_shape[2], 3))

    image_flat = image[brain_mask!= 0]

    # probs
    probs = np.apply_along_axis(lambda x: tissue_model[x,:], 0, np.uint16(image_flat))

    # discretized
    seg = np.argmax(probs, axis=1) + 1

    result_discr[brain_mask!=0] = seg

    for i in range(3):
        result_prb[brain_mask!=0,i] = probs[:,i]

    return result_discr, result_prb

def label_propagation(brain_mask, atlas_list):
    """
    Generate final predicted labels volume from a list of registered 
    probabilistic atlas. The atlasses must be previously registered to a target image.
    Also returns the list of atlasses as a np.ndarray for further usage.

    Args:
        brain_mask (np.ndarray): 3D volume (same size as target image) with brain tissues GT != 0
        atlas_list (List): List np.ndarray volumes, same size as target image

    Returns:
        result_discr (np.ndarray): Discrete segemented brain with labels 1 - CSF, 2 - WM, 3 - GM
        result_prb (np.ndarray): Tissue probabilities volumes. Last axis indexes of array corresponds to
                                probability volumes: 0 - CSF, 1 - WM, 2 - GM
    """
    
    image_shape = atlas_list[0].shape
    result_discr = np.zeros((image_shape))
    result_prb = np.zeros((image_shape[0], image_shape[1], image_shape[2], 3))
    probs = np.zeros(((brain_mask > 0).sum(),3))

    result_prb[:,:,:,0] = atlas_list[0]
    result_prb[:,:,:,1] = atlas_list[1]
    result_prb[:,:,:,2] = atlas_list[2]

    for i in range(3):
        probs[:,i] = result_prb[brain_mask != 0, i]
        result_prb[brain_mask == 0, i] = 0

    # discretized
    seg = np.argmax(probs, axis=1) + 1
    result_discr[brain_mask!=0] = seg

    return result_discr, result_prb