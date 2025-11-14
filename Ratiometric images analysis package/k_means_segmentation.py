# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:30:43 2021

@author: ernes
"""
import numpy as np
import cv2
from skimage import io, filters, measure, feature, morphology, img_as_ubyte, segmentation, color
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage.morphology import disk, binary_opening, binary_closing
from skimage.segmentation import clear_border, watershed



def k_means_seg (img, K=3, attempts=10, show_result=True):
    '''
    This function takes an image, perform k means segmentation algorithm, and return the segmented mask.
    
    Parameters
    --------------
    img : a 2-dimensional array.
        The image to be segmented.
    K : integer (optional).
        The K parameter for kmeans clustering. It defines how many clusters at the end. Default = 3.
    attempts : integer (optional).
        The number of attempts for kmeans clustering.
    show_result : boolean.
        Whether to show the segmentation result.
        
    Returns
    --------------
    res : a 2-dimensional array.
        A labelled and segmented image.
    mask : a boolean.
        A binary mask.
    '''    

    
    #linearize image
    img_linear = img.reshape(-1)

    # convert to float 32
    img_linear = np.float32(img_linear)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                10, 1.0)
    
    ret,label,center = cv2.kmeans(img_linear, K, None, criteria, attempts, 
                                 cv2.KMEANS_PP_CENTERS)
    
    center = np.uint16(center)
    
    # assign a pixel value to each pixel based on center
    res_linear = center[label.flatten()]
    
    res = res_linear.reshape((img.shape))
    
    # 
    labels_stat = measure.regionprops(res)
    
    # if labels_stat
    
    # sort the label objects based on label intensities
    if labels_stat == None:
        res = 0
        mask = 0
        print('no objects were found in the binary mask')
        return res, mask
    
    largest_idx = np.argmin([aCell.label for aCell in labels_stat])
    
    
    # the background ususally has the lowest intensity cluster/label; the higher intensity clusters are features 
    # cluster is the high intensity pixels within the cell    
    inverted_mask = res == labels_stat[largest_idx].label 
    mask = ~inverted_mask 

    if show_result:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        
        ax.imshow(img, cmap='gray_r')
        
        ax.imshow(mask, cmap='Reds', alpha=0.2)
        # contours = measure.find_contours(mask, 0)
        # for acontour in contours:
        #     ax.add_patch(patches.Polygon(acontour[:, [1, 0]],linewidth=1.5,edgecolor='r',facecolor='none', alpha=0.5))
        ax.set_title('Kmeans')
        ax.axis('off')
        plt.show()
    
    return res, mask


def binary_operation(mask, SE=disk(1), small_obj=5000, fill_holes=True, hole_area_thres=200):
    '''
    binary_operation takes a binary mask and apply a series of morpholological operations to refine the mask
    
    Parameters
    --------------
    mask : a boolean
        The binary image to be processed.
    SE : an array
        The structuring element used for the median filter.
    small_obj : an integer
                The size threshold for any given object. Objects smaller than this number will be removed from the mask.
    fill_holes : a boolean
                Whether to fill holes or not.
    hole_area_thres : an integer
                    The size threshold for any given holes. Holes smaller than this number will be filled.
    
    Returns
    --------------
    mask : a boolean.
        The processed binary mask.
    '''
    
    mask = filters.median(mask, footprint=SE)
    
    if fill_holes:
        # mask = ndimage.morphology.binary_fill_holes(mask, structure=disk(1)) # fill holes that are smaller than disk(2)
        mask = morphology.remove_small_holes(mask, area_threshold=hole_area_thres) # fill holes that are smaller than hole_area_thres
    
    mask = ndimage.morphology.binary_erosion(mask, structure=SE, iterations=1)
    
    mask = morphology.remove_small_objects(mask, min_size=small_obj)
    
    
    return mask

def binary_operation_adh(mask, SE=disk(1), small_obj=1000):
    '''
    binary_operation takes a binary mask and apply a series of morpholological operations to refine the mask
    
    Parameters
    --------------
    mask : a boolean.
        The binary image to be processed.
    SE : an array
        The structuring element used for the median filter.
        
    Returns
    --------------
    mask : a boolean.
        The processed binary mask.
    '''
    
    # mask = filters.median(mask, selem=SE)
    
    mask = ndimage.morphology.binary_fill_holes(mask)
    
    # mask = ndimage.morphology.binary_erosion(mask, structure=SE, iterations=1)
    
    mask = ndimage.morphology.binary_dilation(mask, structure=SE, iterations=1)
    
    mask = morphology.remove_small_objects(mask, min_size=small_obj)
    
    return mask

def remove_top_percentile_pix(img, percentile = 95):
    '''
    This function replace the top x percentile of an image with the median and
    the bottom 1% of pixels with 0. 
    This helps the downstream segmentation process identify the cell.
    '''
    clipped_img = img.copy() # make a copy of the original image
    
    top_thres = np.percentile(img, percentile) # determine the top percentile 
    clipped_img[np.where(img >= top_thres)] = np.percentile(img, 50) # that's the median    

    bottom_thres = np.percentile(img, 1) # for image that's been background subtracted, this probably won't do anything.
    clipped_img[np.where(img <= bottom_thres)] = 0 # convert the lowest 1% pixels into 0
    

    return clipped_img

def watershed_seg(img, binary, kernel_size=3):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    opening = binary_opening(binary, footprint=kernel)
    opening = clear_border(opening) #Remove edge touching grains
    
    dist_transform = ndimage.distance_transform_edt(opening)
    local_maxima = feature.peak_local_max(dist_transform, exclude_border = False, threshold_rel=dist_transform.max()*0.0064)
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[tuple(local_maxima.T)] = True
    markers = measure.label(mask)


    watershed_img = watershed(-dist_transform, markers, connectivity=8, mask=img) 
    plt.imshow(watershed_img)
    plt.show()
    
    return watershed_img









    
    






