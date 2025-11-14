#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:56:04 2022

@author: ernestiu
"""
import sys
sys.path.append('//Ratiometric images analysis package/')
import numpy as np
import matplotlib.pyplot as plt
import os
from proprietary_image_readers import nd2_image_reader
import tifffile
from image_corr_module import bg_subtraction

image_path = '/image_folder/'

bg_value_list = {} # create a dictionary 
    
img_name_list = []

n = 0


for nd2 in sorted(os.listdir(image_path)):  #iterate through each file  
    if nd2.endswith('.nd2') and not nd2.startswith('.'): #'shading image'
        ### import nd2 here ###    
        # image_stack dimensions: fov,frame,x,y,ch,z
        image_stack, fields_of_view, image_name, image_metadata = nd2_image_reader(image_path + nd2) 
        pixel_size = image_metadata['pixel_microns']
        print('Processing: ' + str(image_name))
        
        if len(image_stack.shape) <6:
            image_stack = np.expand_dims(image_stack, axis=5) # add a fake z axis if there's no z 
            
        img_name_list.append(image_name)
        
        bg_value_list[str(image_name)] = {} # create a nested dictionary
        
        for fov in fields_of_view:
            
            if fov != 0: print('Processing FOVs: {0}/{1}'.format(fov+1, len(fields_of_view))) # only print the process if fov is larger than 1
            
                
            final_image = np.empty((image_stack.shape[0],image_stack.shape[1],image_stack.shape[2],
                                    image_stack.shape[3],image_stack.shape[4], 1), dtype=image_stack.dtype) 
            
            
            
            for ch in range(image_stack.shape[4]): # for every channel
                
                if image_stack.shape[5] > 1: # if it's a z-stack
                    actin = np.max(image_stack[:,:,:,:,ch,:],axis=4) # max projection
                    print('z maximum projection applied!')
                else:
                    actin = image_stack[:,:,:,:,ch,0]
                
                bg_values = []
                for oo in range(final_image.shape[1]): # for each time point
                    
                
                    final_image[fov,oo,:,:,ch,0], bg_value_1 = bg_subtraction(actin[fov,oo,:,:], show_image = oo==0, bg_value = None)

                    bg_values.append(bg_value_1)

                bg_value_list[str(image_name)]['Channel_' + str(ch)] = bg_values              
                
                # save it 
                def append_str(str_1, str_2, append = False):
                    '''
                    This simple function appends a string (str_2) to another string (str_1) based on condition. 
                    If append is True, the function will combine str_1 and str_2, and return a combined string.
                    '''
                    if append == True:
                        return str_1 + str_2
                    else: 
                        return str_1
                
                final_image_name = append_str(str_1 = os.path.dirname(image_path) + '/' + image_name +'_ch' + str(ch+1) + '_bg_subtracted', 
                                              str_2 = '_fov_' + str(fov+1), append = len(fields_of_view)>1)#fov!=0)
                
                tiff_metadata = {'unit': 'micron'}

                tifffile.imwrite(final_image_name + '.tif', final_image[fov,oo,:,:,ch,0], imagej=True,
                                  resolution=(1/pixel_size, 1/pixel_size), metadata=tiff_metadata)
                
                n += 1


cmap = plt.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0, 1, image_stack.shape[4])]

channel_no = image_stack.shape[4]

for ch in range(image_stack.shape[4]):
    channel_intensities_list = [] 
    for key in bg_value_list.values():
        channel_intensities_list.append(key['Channel_' + str(ch)][0])
    plt.plot(channel_intensities_list, color = colors[ch], label='Channel ' + str(ch+1))

plt.xticks(range(len(bg_value_list)), img_name_list, rotation=90)
plt.legend()
plt.show()
        
