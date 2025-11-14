# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:47:13 2020

@author: ernes
"""
import numpy as np
from skimage import io, filters, restoration, exposure, morphology
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, argrelextrema
import os
from skimage.util import img_as_float, img_as_uint
from scipy.signal import convolve2d
import tifffile

def remove_empty_frames(img, axis_to_remove_images = 0, indices = None):
    if indices:
        filtered_img = np.delete(img, [indices], axis=axis_to_remove_images)
        
    else:
        max_list = [np.max(img[i,:,:]) for i in range(img.shape[0])]
        indices = [i for i, x in enumerate(max_list) if x == 0]    
        filtered_img = np.delete(img, [indices], axis=axis_to_remove_images)
    print('Frames dropped: {}. A total of {} frames were discarded.'.format(str(indices), len(indices)))
    
        
    return filtered_img
    
    
def uneven_corr(shading_img, img, psudo_bg = False):
    
    if psudo_bg == False: 
        img_heterogenity = np.std(shading_img)/np.mean(shading_img)
        if img_heterogenity > 0.025: 
            bg = filters.gaussian(shading_img, sigma=2, preserve_range=True)
            # print('uneven_corr: Gaussian filter was applied to make the shading image more homogenous.')
    
        else: # if the image is already pretty homogenous, don't apply the gaussian filter
            print('Shade image heterogenity: ' + str(round(img_heterogenity, 4)))
            print('uneven_corr: The image is already homogenous. No gaussian filter was applied.')
            bg = shading_img
        
        # plt.imshow(bg)
        # plt.show()
    
    else: # using a blurred original image as a psudo background image
        bg = filters.gaussian(img, sigma=150, preserve_range=True)
    # normalize the background to 1
    bg = bg/np.mean(bg)
    

    corr_img = img/bg
        
    return corr_img
    
    


def d_exponential(x, a, b, c, d):
    '''
    The double exponential equation.
    '''
    return (a*np.exp(b*x) + c*np.exp(d*x))


def photobleaching_corr(img, thresholded_img = False, title = None):
    
    '''
    Description
    ------------
    Photobleaching correction. 1. Smoothed Double exponential 2. Raw Double exponential 3. Polynomial 
    
    Parameters
    ------------
    img : a 3-dimensional array.
        The timelapse image to be corrected.
        
    thresholded_img : boolean.
                    If True, it will take only the true pixels into considerations, e.g., a segmented/masked image with background = 0.
                    If False, it will use triangle thresholding to estimate the feature mask.
    
    Returns
    ------------
    corr_img : a 3-dimensional array.
        The photobleaching corrected image.
    '''
    
    # measure the mean intensity across frames - only measure True pixels
    sum_series = []
    all_locs = [] 
    
    for i in range (img.shape[0]):
        if thresholded_img:
            locs = np.where(img[i,:,:] > 0)

        else: 
            cell_binary = img[i,:,:] > filters.threshold_triangle(img[i,:,:]) # do a rough threshold to select the true pixels
            
            if i == 0: # show the segmentation of the first frame for bleaching correction
                plt.imshow(img[i,:,:], cmap='gray_r')
                plt.imshow(~cell_binary, cmap='Reds', alpha=0.3)
                plt.title('Photobleaching Correction: Red = Background')
                plt.show()
                
            locs = np.where(cell_binary == True)

        pixels = img[i,:,:][locs]
        sum_series.append(np.sum(pixels)) # append the sum of true pixels
        all_locs.append(locs) # append the positions of true pixels

    window_size = int(np.ceil((len(sum_series)/2)) // 2 * 2 + 1) # divide number of frames by 1.6 and round up to an odd integer
    smoothed_y = savgol_filter(sum_series, window_size, 1) #3 order
    
    #mean_series = [np.mean(img[i,:,:]) for i in range (img.shape[0])]
    # local_minima_idx = argrelextrema(np.array(mean_series), np.less)
    
    
    # create a list with number of frames
    x_axis = np.linspace(start=0, stop=len(sum_series)-1, num=len(sum_series))
    
    
    try:
        # calculate the fitting parameters
        popt, pcov = curve_fit(f=d_exponential, xdata=x_axis, ydata=smoothed_y, p0=[0,0,0,0])
        fit_y = d_exponential(x_axis, *popt) # fit the data
        plt.scatter(x_axis, smoothed_y, linewidth=1, color='blue', label='Smoothed intensity')
        
    except:
        # if smoothed data did not work, try with the raw data
        try: 
            popt, pcov = curve_fit(f=d_exponential, xdata=x_axis, ydata=sum_series, p0=[0,0,0,0])
            fit_y = d_exponential(x_axis, *popt) # fit the data
            print('****Smoothed data did not work. Raw data was used for curve fitting****')
            
        except:
            try: 
                poly_model = np.poly1d(np.polyfit(x_axis, smoothed_y, 3)) # try polynomial fit
                fit_y = poly_model(x_axis) # fit the data
                print('****Polynomial fit was used for curve fitting****')
            except:
                # if none of the fitting algorithm works, just return the original img
                plt.scatter(x_axis, sum_series, linewidth=1, color='red', label='Raw intensity')
                # plt.scatter(x_axis, smoothed_y, linewidth=1, color='blue', label='Smoothed intensity')
                plt.xlabel("Frame")
                plt.ylabel("Intensity, a.u.")
                plt.legend()
                plt.show()
                print("****Photobleaching correction unsuccessful!!****")
                
                return img
                
    
    plt.scatter(x_axis, sum_series, linewidth=1, color='red', label='Raw intensity')
    plt.plot(x_axis, fit_y, linestyle='--', linewidth=2, color='black', label='Fitted data')

    coefficients = [sum_series[0]/fit_y[i] for i in range(len(fit_y))]

    corr_img = np.empty(img.shape, dtype=img.dtype)
    new_sum_list = []
    
    for i in range(corr_img.shape[0]):
        corr_img[i] = img[i]*coefficients[i]
        pixels = corr_img[i,:,:][all_locs[i]]
        new_sum_list.append(np.sum(pixels))
    
              
    plt.plot(x_axis, new_sum_list, linestyle='dashdot', linewidth=2, color='blue', label='Corrected intensity')
    
    plt.xlabel("Frame")
    plt.ylabel("Intensity, a.u.")
    plt.title(title)
    plt.legend()
    plt.show()
    
    return corr_img
                    
def rolling_ball_background_subtract(img, r=100):
    '''
    Description
    ------------
    This function performs a rolling-ball background subtraction.
    
    Parameters
    ------------
    img: a 2D array
    r: radius of the rolling ball
    
    Returns
    ------------
    background_subtracted: a 2D array after background subtraction
    
    '''
    background = restoration.rolling_ball(img, radius=r)
    background_subtracted = img - background
    return background_subtracted

def bg_subtraction(original_img, show_image = False, method = 'triangle', bg_value = None):
    
    if bg_value == None: # if a background value is not provided
        if method == 'triangle': # approximate the background using triangle
            try:
                p99 = np.percentile(original_img, (99))
                filtered_original_img = original_img[original_img < p99]
                threshold_val = int(filters.threshold_triangle(filtered_original_img))
                bg = original_img > threshold_val 


            except:
                print('Cannot determine the threshold for background subtraction')
                return original_img, 0
        if method == 'otsu':
            try: 
                p99 = np.percentile(original_img, (99))
                filtered_original_img = original_img[original_img < p99]
                threshold_val =int(filters.threshold_otsu(filtered_original_img))
                bg = original_img > threshold_val
            except:
                print('Cannot determine the threshold for background subtraction')
                return original_img, 0

    
        bg_locs = np.where(bg == False) 
          
        bg_value = int(np.median(original_img[bg_locs])) # calculate the median of background values
    
    else:
        bg = original_img > bg_value
    
    subtracted_img = original_img
    
    # increase the value of pixels that are lower than the background to avoid integer underflow
    underflow_pix_loc = np.where(subtracted_img < bg_value) # locate all the underflow pixels
    
    subtracted_img = subtracted_img - bg_value # subtract background from the image
    
    subtracted_img[underflow_pix_loc] = 0

    # print('Background value (a.u.): ' + str(bg_value)) 
    
    if show_image == True:
        fig, ax = plt.subplots(1,2, figsize=(10, 5), dpi=150)
        ax[0].imshow(original_img, cmap='Blues')#cmap='gray_r')
        ax[0].imshow(~bg, cmap='Reds', alpha=0.3)
        ax[0].set_title('Background (red) selected for substraction')
        ax[0].axis('Off')
        
        
        ax[1].hist(original_img.ravel(), bins=50)
        ax[1].set_yscale('log')
        ax[1].set_ylabel('Fraction (log)')
        ax[1].axvline(bg_value, color='r')
        # plt.axvline(np.median(filtered_original_img), color='b')
        plt.show()

    return subtracted_img.astype(original_img.dtype), bg_value # 
    

def bleedthru_corr(yfp, fret, cfp, yfp_coeff = 0.0019, cfp_coeff = 0.0739, show_image = False):

    yfp_bleedthru = yfp*yfp_coeff # bleedthru from yfp
    
    cfp_bleedthru = cfp*cfp_coeff # bleedthru from cfp
    
    bleedthru_img = yfp_bleedthru+cfp_bleedthru # sum of the bleedthrough
    
    # increase the value of pixels that are lower than the background to avoid integer underflow
    underflow_pix_loc = np.where(fret < bleedthru_img) # locate all the underflow pixels
    # fret[underflow_pix_loc] = fret[underflow_pix_loc] + (bleedthru_img[underflow_pix_loc] - fret[underflow_pix_loc])

    final_corrected_img = (fret-bleedthru_img).astype(fret.dtype)
    
    final_corrected_img[underflow_pix_loc] = 0
    
    
    if show_image == True:
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,30))
        ax1.imshow(cfp_bleedthru)
        ax1.set_title('CFP Bleedthrough Image')
        ax1.axis('off')
        ax2.imshow(yfp_bleedthru)
        ax2.set_title('YFP Bleedthrough Image')
        ax2.axis('off')
        ax3.imshow(final_corrected_img)
        ax3.set_title('Bleedthrough corrected FRET Image')
        ax3.axis('off')
        plt.show()
        
    return final_corrected_img

def hybrid_median_filtering(img):
        
    kernel_diag = np.array([[1,0,0,0,1],
                            [0,1,0,1,0],
                            [0,0,0,0,0],
                            [0,1,0,1,0],
                            [1,0,0,0,1]], dtype='uint8')
    
    kernel_vnh = np.array([[0,0,1,0,0],
                           [0,0,1,0,0],
                           [1,1,0,1,1],
                           [0,0,1,0,0],
                           [0,0,1,0,0]], dtype='uint8')
    
    img_diag_filt = filters.median(img, kernel_diag, mode='constant')
    
    img_vnh_filt = filters.median(img, kernel_vnh, mode='constant')
    
    hmf_img = np.median([img_vnh_filt,img_diag_filt,img], axis=0) 
    

    
    return hmf_img.astype(img.dtype)


def remove_spurious_pixels(img, threshold = 1000):
    '''
   
    This function replaces spurious pixels that exceed a threshold with a median of its surrounding.
    In FRET analysis, a number that exceeds 100% FRETeff can't exist. Therefore, we need to correct those pixels
    with a number, which, in this case, could be the median of its neighbors. 
    
    '''           
    new_img = img.copy()
    spurious_pixels_loc = np.where(img > threshold) # the locations of the spurious pixels
    
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype='uint8')
    
    median_img = filters.median(img, kernel, mode='constant')
    new_img[spurious_pixels_loc] = median_img[spurious_pixels_loc]   
    if len(spurious_pixels_loc[0]) > 0: 
          print('remove_spurious_pixels: Pixels that exceed the limit (' + str(threshold) + '): ' + str(len(spurious_pixels_loc[0])))
    return new_img
    
    

