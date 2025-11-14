# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:30:07 2021

@author: ernes
"""
from skimage import morphology, filters
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import tifffile



def kymo_to_coords(kymo, thres=15, pixel_length = 1):
    """
    This function takes a kymograph and extract the membrane position from the image.
    *** make sure the kymo is oriented in the right orientation***
    Parameters
    ----------
    kymo : array
        A kymograph.
    thres : integer, optional
        The largest membrane displacement allowed in a frame. The default is 15 pixels.
    pixel_length : integer, optional
        The pixel size. The default is 0.1833333.

    Returns
    -------
    normalized_coords : list
        A list of coordinates in tuples where the lowest point is normalized to 0.
    filtered_coords : list
        A list of coordinates in tuples where extremas are filtered and replaced by the interpolated position between frames.

    """
    
    smooth_kymo = filters.median(kymo > 0, morphology.disk(1))
    
    smooth_kymo = morphology.binary_erosion(smooth_kymo, morphology.rectangle(5,1))
    
    # io.imsave('C:/Users/ernes/Desktop/test/testing_3_processed.tif', smooth_kymo)
    # plt.imshow(smooth_kymo)
    # plt.show()
    sobel_kymo = filters.sobel(smooth_kymo)
    # plt.imshow(sobel_kymo)
    # plt.show()
    coords = []
    coords.append(np.argmax(np.gradient(smooth_kymo[:,0]*1)))
    for time in range(1, sobel_kymo.shape[1]-1):
        local_max = np.argmax(sobel_kymo[:,time])
        coords.append(local_max)
    coords.append(np.argmax(np.gradient(smooth_kymo[:,-1]*1)))
    
    # the following code normalize the coordinates to the lowest point
    normalized_coords = []
    for jj in range(len(coords)):
        #norm_coords = filtered_coords[jj]*-1 + max(filtered_coords)
        norm_coords = (coords[jj]*-1 + max(coords)) * pixel_length
        normalized_coords.append(norm_coords)
    
    return np.array(normalized_coords)


def ccf_values (series_1, series_2):
    p = (series_1 - np.mean(series_1))/(np.std(series_1) * len(series_1))
    q = (series_2 - np.mean(series_2))/np.std(series_2)
    c = np.correlate(p, q, 'full')
    return c

def ccf_plot(lags, ccf):
    
    from scipy.interpolate import splev, splrep
    tck = splrep(lags, ccf, s=0.5)
    y_spline = splev(lags, tck, der=0)
    # spl = splrep(lags, ccf)
    # y_spline = splev(lags, spl)

    
    figure, axes =plt.subplots(figsize=(9, 6))
    axes.plot(lags, ccf, c='blue', linewidth=3)
    axes.plot(lags, y_spline, c='red', linewidth=3)
    # ax.axhline(-2/np.sqrt(23), color='red', label='5% confidence interval')
    # ax.axhline(2/np.sqrt(23), color='red')
    axes.axvline(x = 0, color = 'black', lw = 1)
    axes.axhline(y = 0, color = 'black', lw = 1)
    # ax.axhline(y = np.max(ccf), color = 'blue', lw = 1, 
    # linestyle='--', label = 'highest +/- correlation')
    # ax.axhline(y = np.min(ccf), color = 'blue', lw = 1, 
    # linestyle='--')
    axes.set(ylim = [-0.7, 0.7])
    axes.set_title('Cross Correation', weight='bold', fontsize = 15)
    axes.set_ylabel('Correlation Coefficients', weight='bold', fontsize = 12)   
    axes.set_xlabel('Time Lags', weight='bold', fontsize = 12)
    # plt.legend()
    plt.show()
    


img_path = '/image_folder/'



frame_interval = 10 # second

# I multiplied the pixel size by 3 because every pixel on the kymo graph is 3 pixels collapsed/averaged into 1 pixel in the kymo 
pixel_size = 0.2277115*3  # micron 
pixel_size *= 1000 # convert to nm
length = 3 # the length of lamellipodia in pixel - roughly translate to 1.5 um

num = 0
img_name = []
all_lags = []
all_corrs = []
all_gc_matrices = []

for n, image in enumerate(sorted(os.listdir(img_path))):  #iterate through each file 

    if image.endswith('.tif'):
        
        image_name = os.path.splitext(image)[0]
        img_name.append(image_name)
        
        img = tifffile.imread(img_path + image)
        
        img_rot = np.fliplr(np.rot90(img,k=3))
        
        # how many time frame should we consider e.g. consider only from frame 25 to the final frame would be range(25, img_rot.shape[1])
        threshold_range = range(0, img_rot.shape[1])  

        normalized_coords = kymo_to_coords(img_rot, pixel_length = pixel_size)

        
        instant_velo = np.gradient(normalized_coords) / frame_interval # convert from frame to second
        
        normalized_coords = normalized_coords[threshold_range]
        instant_velo = instant_velo[threshold_range]
        
        
        
        threshold = 0.1 
        
        protrusion_idx = []
        stall_idx = []
        retract_idx = []
        LP_activity = []
        
        for each_point in range(len(normalized_coords)-1):
            difference = normalized_coords[each_point+1] - normalized_coords[each_point]
            if difference < -threshold: retract_idx.append(each_point), LP_activity.append(-1)
            elif difference > threshold: protrusion_idx.append(each_point), LP_activity.append(+1)
            else: stall_idx.append(each_point), LP_activity.append(0)
        

        
        calcium_intensities = []


        
        for i in threshold_range: #img_rot.shape[1]
            
            intensity_line = img_rot[1::,i][img_rot[1::,i] != 0][1::] # remove the first pixel
            lamellipodia_ca_int = np.mean(intensity_line[0:length]) # measure mean of x number of pixels away from cell edge
            calcium_intensities.append(lamellipodia_ca_int)

            
        calcium_intensities = np.array(calcium_intensities)
        
         
        calcium_intensities = gaussian_filter1d(calcium_intensities, 1)
        instant_velo = gaussian_filter1d(instant_velo, 1)
        normalized_coords = gaussian_filter1d(normalized_coords, 1)

        
        normalized_coords = np.array(normalized_coords) # convert list to array

        
        plt.scatter(y=normalized_coords[protrusion_idx], x=protrusion_idx, color='g')
        plt.scatter(y=normalized_coords[stall_idx], x=stall_idx, color='y')
        plt.scatter(y=normalized_coords[retract_idx], x=retract_idx, color='r')
        plt.ylabel('Edge position')
        plt.xlabel('Frames')
        plt.title(image_name)
        plt.show()
    
        fig,ax = plt.subplots()
        ax.plot(instant_velo)
        fig.suptitle(image_name)
        ax.set_ylabel('Instantaneous velocity', color='b')
        ax.set_xlabel('Frames')
        ax2=ax.twinx()
        ax2.plot(calcium_intensities, color='r')
        ax2.scatter(y=calcium_intensities[protrusion_idx], x=protrusion_idx, color='g')
        ax2.scatter(y=calcium_intensities[stall_idx], x=stall_idx, color='y')
        ax2.scatter(y=calcium_intensities[retract_idx], x=retract_idx, color='r')
        ax2.set_ylabel('Ca2+ level in lamellipodia', color='r')
        
        plt.show()
        

        fig,ax = plt.subplots()
        ax.plot(normalized_coords)
        fig.suptitle(image_name)
        ax.set_ylabel('Edge position', color='b')
        ax.set_xlabel('Frames')
        ax2=ax.twinx()
        ax2.plot(calcium_intensities, color='r')
        ax2.scatter(y=calcium_intensities[protrusion_idx], x=protrusion_idx, color='g')
        ax2.scatter(y=calcium_intensities[stall_idx], x=stall_idx, color='y')
        ax2.scatter(y=calcium_intensities[retract_idx], x=retract_idx, color='r')
        ax2.set_ylabel('Ca2+ level in lamellipodia', color='r')
        
        plt.show()

        num += 1
        print('Analyzed cell {0}'.format(num))
        
            
    
        ccf = ccf_values(calcium_intensities, normalized_coords) 
        
        lags = signal.correlation_lags(len(calcium_intensities), len(normalized_coords)) 
        # the first variable is the X and the variable that is being shifted.
        # The coefficient values on the left of zero are those where X leads and Y lags while the ones on the right are when Y leads and X lags.
          
        ccf_plot(lags, ccf)

        
        
        time = np.linspace(0,(len(normalized_coords)-1)*frame_interval, num=len(normalized_coords), dtype=int)
        
        dataframe = pd.DataFrame()
        
        dataframe['Time'] = pd.Series(time)
        dataframe['Edge Position'] = pd.Series(normalized_coords)
        dataframe['Instantaneous Velocity'] = pd.Series(instant_velo)
        dataframe['LP Ca2+'] = pd.Series(calcium_intensities)
        dataframe.set_index('Time')
        
        LP_activity_df = pd.DataFrame(data=LP_activity, columns=['LP activity'])
        
        CCF_df = pd.DataFrame(data=zip(lags, ccf), columns=['Lags', 'CCFs'])
        
        dataframe = pd.concat([dataframe,LP_activity_df], axis=1) 
        
        dataframe = pd.concat([dataframe,CCF_df], axis=1) 

        dataframe.to_excel(img_path + image_name + '_data.xlsx', engine='xlsxwriter', index=False)  


    