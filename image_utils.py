import cv2
import os
import numpy as np

#os.system('pip install -U efficientnet==0.0.4') #Used from Script Utility https://www.kaggle.com/ratthachat/efficientnet
#from efficientnet import preprocess_input

RANGE_0_1 = 1
RANGE_1_1 = 2
RANGE_0_255 = 3

TRANSFORM_NONE = 0
TRANSFORM_0_1_to_0_255 = 1
TRANSFORM_1_1_to_0_255 = 2
TRANSFORM_0_255_to_0_1 = 3
TRANSFORM_0_255_to_1_1 = 4
TRANSFORM_1_1_to_0_1 = 5
TRANSFORM_0_1_to_1_1 = 6

def crop_image_from_gray(img, tol=7):
    """ Crop image which have values above tol in gray scale """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def check_range(image):
    """ Check range of values of the image """
    if image.min() < 0:
        return RANGE_1_1
    elif image.max() > 1:
        return RANGE_0_255
    else:
        return RANGE_0_1

def transform_range(image, flag):
    """ Transform image with the specified transformation """
    if flag == TRANSFORM_NONE:
        return image
    elif flag == TRANSFORM_0_1_to_0_255:
        image = image * 255
        return np.round(image).astype(np.uint8)
    elif flag == TRANSFORM_1_1_to_0_255:
        image = (image + 1) * 127.5
        return np.round(image).astype(np.uint8)
    elif flag == TRANSFORM_0_255_to_0_1:
        image = image / 255.
        return image.astype(np.float32)
    elif flag == TRANSFORM_1_1_to_0_1:
        image = (image / 2.) + 1.
        return image.astype(np.float32)
    elif flag == TRANSFORM_0_1_to_1_1:
        image = (image * 2.) - 1.
        return image.astype(np.float32)

def preprocess_image(image, sigmaX=2):
    """
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness
    """
    if image.ndim == 2:
        img_size = image.shape
    elif image.ndim == 3:
        channel_idx = np.argmin(image.shape)
        img_size = image.shape[1:] if channel_idx == 0 else image.shape[0:2]
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = crop_image_from_gray(image)
    image = cv2.resize(image, img_size)
    #image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #image = preprocess_input(image)
    return image