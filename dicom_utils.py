import numpy as np
import pandas as pd
import cv2
import pydicom
from matplotlib import pyplot as plt
from scipy import interpolate
import warnings
import os

from tqdm import tqdm_notebook

# From https://radiopaedia.org/articles/windowing-ct
dicom_windows = {
    'brain' : (80,40),
    'subdural':(200,80),
    'stroke':(8,32),
    'brain_bone':(2800,600),
    'brain_soft':(375,40),
    'lungs':(1500,-600),
    'mediastinum':(350,50),
    'abdomen_soft':(400,50),
    'liver':(150,30),
    'spine_soft':(250,50),
    'spine_bone':(1800,400)
}

def crop_brain(image, n_top_areas = 5, max_scale_diff = 3, plot = False, dimensions = None, area = None):
    image = normalize_img(image, use_min_max = True)
    if dimensions is None:
        if (image.max() - image.min()) == 0 or np.isnan(image.max()) or np.isnan(image.min()):
            raise ValueError('Empty image')
        gray = np.uint8(image * 255)
        blur = cv2.blur(gray, (5, 5)) # blur the image
        # Detect edges using Canny
        #canny_output = cv2.Canny(blur, threshold, threshold * 2)
        # Find contours
        contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Cycle through contours and add area to array
        areas = []
        for c in contours:
            areas.append(cv2.contourArea(c))

        # Sort array of areas by size
        sorted_areas = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)
        biggest_area = sorted_areas[0][0]

        # Approximate contours to polygons + get bounding rects and circles
        contours_poly = []
        boundRect = []
        min_dist_to_center = np.inf
        best_contour_idx = 0
        for i, c in enumerate(sorted_areas):
            # Only treat contours which are in top 5 and less than 'max_scale_diff' times smaller than the biggest one
            if c[0] > 0 and i < n_top_areas and biggest_area/c[0] < max_scale_diff:
                contour_poly = cv2.approxPolyDP(c[1], 3, True)
                contours_poly.append(contour_poly)
                boundRect.append(cv2.boundingRect(contour_poly))
                center, _ = cv2.minEnclosingCircle(contour_poly)

                # Calculate distance from contour center to center of image
                dist = (gray.shape[0]//2 - center[0])**2 + (gray.shape[1]//2 - center[1])**2
                if min_dist_to_center > dist:
                    best_contour_idx = i
                    min_dist_to_center = dist
            else:
                break

        # Get boundaries of the Rectangle which includes the contour
        x,y,w,h = boundRect[best_contour_idx]
        best_area = sorted_areas[best_contour_idx]
    else:
        x,y,w,h = dimensions
        best_area = area
    # Crop the image
    cropped = image[y:y+h,x:x+w]
    # Pad needed pixels
    final_image = pad_square(cropped)
    
    # Show three images (original, cropped, final)
    if plot:
        fig=plt.figure(figsize  = (10,30))    
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(image)
        ax.add_patch(patches.Rectangle(
            (x, y),
            w,
            h,
            fill=False      # remove background
         )) 
        fig.add_subplot(1, 3, 2)
        plt.imshow(cropped)
        fig.add_subplot(1, 3, 3)
        plt.imshow(final_image)
        plt.show()
    return final_image, best_area, (x,y,w,h)

def pad_square(x):
    r,c = x.shape
    d = (c-r)/2
    pl,pr,pt,pb = 0,0,0,0
    if d>0: pt,pd = int(np.floor( d)),int(np.ceil( d))        
    else:   pl,pr = int(np.floor(-d)),int(np.ceil(-d))
    return np.pad(x, ((pt,pb),(pl,pr)), 'minimum')

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def get_rows_columns(data):
    dicom_fields = [data[('0028', '0010')].value, #Rows
                    data[('0028', '0011')].value] #Columns
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def get_subgroups(data):
    # There are 3 groups, but we will create a fourth for Others:
    # 1) Bits Stored 16bits
    # 2) Bits Stored 12bits - Pixel Representation 0
    # 3) Bits Stored 12bits - Pixel Representation 1
    # -1) Others (in case new data appears)
    dicom_fields = [data[('0028', '0101')].value, #Bits Stored
                    data[('0028', '0103')].value] #Pixel Representation
    dicom_values = [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
    if dicom_values[0] == 16:
        return 1
    elif dicom_values[0] == 12 and dicom_values[1] == 0:
        return 2
    elif dicom_values[0] == 12 and dicom_values[1] == 1:
        return 3
    else:
        return -1

# According to https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000
    return dcm.pixel_array, dcm.RescaleIntercept

def get_freqhist_bins(dcm_img, n_bins = 100):
    imsd = np.sort(dcm_img.reshape(-1))
    t = np.concatenate([[0.001],
                       np.arange(n_bins).astype(np.float64)/n_bins+(1/2/n_bins),
                       [0.999]])
    t = (len(imsd)*t).astype(np.int64)
    return np.unique(imsd[t])

def get_dcm_img(path, window_type = 'brain', verbose = True):
    # Read and scale of DICOM images according to its metadata
    dcm = pydicom.dcmread(path)
    window_center, window_width, intercept, slope = get_windowing(dcm)
    group = get_subgroups(dcm)   
    
    if group == 2 and (int(intercept) > -100):
        dcm_img, intercept = correct_dcm(dcm)
        dcm_img = dcm_img * slope + intercept
    else:
        dcm_img = get_pixel_array(dcm, path)  * slope + intercept
    
    window_center = dicom_windows[window_type][1]
    window_width = dicom_windows[window_type][0]
    min_px = window_center - window_width//2
    max_px = window_center + window_width//2
    dcm_img[dcm_img<min_px] = min_px
    dcm_img[dcm_img>max_px] = max_px
    
    if (dcm_img.max() - dcm_img.min()) == 0:
        dcm_img[:, :] = 0
        if verbose:
            warnings.warn('Empty image from path: ' + path, UserWarning)
    
    return dcm_img, group

def get_pixel_array(dcm, path=None):
    try:
        return dcm.pixel_array.astype(np.float32)
    except ValueError as e:
        rows, columns = get_rows_columns(dcm)
        if path is None:
            print("DICOM pixel data could not be extracted due to: ", e)
        else:
            print(f"DICOM pixel data could not be extracted from {path} due to: ", e)
        return np.zeros((rows, columns), dtype=np.float32)

def interpolate_img(dcm_img, bins = None, n_bins = 100):
    # Equal distribution of intensity
    if bins is None: 
        bins = get_freqhist_bins(dcm_img, n_bins)
    
    return np.clip(interpolate.interp1d(bins, np.linspace(0., 1., len(bins)), fill_value="extrapolate")(dcm_img.flatten()).reshape(dcm_img.shape), 0., 1.)

def normalize_img(dcm_img, mean = None, std = None, use_min_max = False):
    # Normalization to zero mean and unit variance
    if use_min_max:
        img_max = dcm_img.max()
        img_min = dcm_img.min()
        if (img_max - img_min) != 0:
            return (dcm_img - img_min) / (img_max - img_min)
        else:
            return dcm_img
    else:
        if mean is None: 
            mean = dcm_img.mean()

        if std is None: 
            std = dcm_img.std()
        return (dcm_img - mean) / std
    

def preprocess_dicom(path, x, y, bins = None, n_bins = 100, mean = None, std = None, use_min_max = False, remove_empty = False, windows_type = 'brain', verbose = True): 
    area = 0
    dimensions = None
    if not type(windows_type) is list:
        windows_type = [windows_type,]
        
    final_dcm_img = np.empty((x,y,len(windows_type)))
    for i,window_type in enumerate(windows_type):
        dcm_img, group = get_dcm_img(path, window_type, verbose)

        # Crop image to show only the brain part (only posible if the image is not empyt)
        try:
            isEmpty = False
            dcm_img, area, dimensions = crop_brain(dcm_img, area = area, dimensions= dimensions)
        except ValueError as e:
            isEmpty = True
            area = 0
            if verbose:
                print("DICOM image from ", path, " is not cropped because gave the following error: ", e)
        finally:
            if isEmpty and remove_empty:
                return None, None
            else:
                # If distributed by groups (different than -1) then use only the values of the group
                if group != -1:
                    if type(bins) == dict:
                        bins = bins[group]
                    if type(mean) == dict:
                        mean = mean[group]
                    if type(std) == dict:
                        std = std[group]

                if not isEmpty:
                    try:
                        dcm_img = interpolate_img(dcm_img, bins, n_bins)
                    except ValueError as e:
                        if verbose:
                            print("DICOM image from ", path, " is not bin interpolated because gave the following error: ", e)
                dcm_img = normalize_img(dcm_img, mean, std, use_min_max)

                # Rescale to the defined image size
                if dcm_img.shape != (x, y):
                    dcm_img = cv2.resize(dcm_img, (x, y), interpolation=cv2.INTER_NEAREST)
            final_dcm_img[:,:,i] = dcm_img
    return final_dcm_img, area

# Samples from each group are extracted by trying to find an specific number of samples per group
# This is done due to memory limitations of have all training metadata in a DataFrame
def sample_groups(load_dir, samples_per_group = 5, max_trys = 1000):
    filenames = os.listdir(load_dir)
    filenames_groups = {1 : [], 2 : [], 3 : []}
    for group in tqdm_notebook([1,2,3], desc = 'Group sample'):
        count_samples = 0
        for _ in tqdm_notebook(range(max_trys), desc = 'Try'):
            filename = np.random.choice(filenames, size = 1, replace = False)[0]
            sample_group = get_subgroups(pydicom.dcmread(load_dir + filename))
            if sample_group == group:
                filenames_groups[group].append(load_dir + filename)
                count_samples += 1
                if count_samples >= samples_per_group:
                    break
    return filenames_groups

# For each group it is computed:
# Firstly mean of equally distributed bin
# Secondly mean of mean pixels values and mean of std pixel values using the previous bin mean
def sample_bins_mean_std(load_dir, samples_per_group = 5, max_trys = 1000, n_bins = 100):
    bins_mean = {}
    mean = {}
    std = {}
    groups_paths = sample_groups(load_dir, samples_per_group = samples_per_group, max_trys = max_trys)
    for group in [1,2,3]:    
        # Do not proceed if there is no images
        if len(groups_paths[group]) == 0:
            bins_mean[group] = []
            mean[group] = np.nan
            std[group] = np.nan
            continue
                    
        filenames = groups_paths[group]
    
        dcm_img_array = []
        for filename in tqdm_notebook(filenames, desc = 'Calc bins'):
            dcm_img, _ = get_dcm_img(filename)
            dcm_img_array.append(dcm_img)
            #bins_array.append(get_freqhist_bins(dcm_img, n_bins))

        #bins_mean[group] = np.array(bins_array).mean(axis = 0)
        bins_mean[group] = get_freqhist_bins(np.array(dcm_img_array).reshape(-1), n_bins)
    
        dcm_img_array = []
        for filename in tqdm_notebook(filenames, desc = 'Calc mean & std'):
            dcm_img, _ = get_dcm_img(filename)
            dcm_img_array.append(interpolate_img(dcm_img, bins_mean[group], n_bins))
    
        mean[group] = np.array(dcm_img_array).flatten().mean()
        std[group] = np.array(dcm_img_array).flatten().std()
    
    return bins_mean, mean, std