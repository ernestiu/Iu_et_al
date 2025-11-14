#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 22:05:14 2022

@author: ernestiu
"""
from nd2reader import ND2Reader
from readlif.reader import LifFile
import os
import numpy as np
import matplotlib.pyplot as plt
import nd2


def lif_image_reader(image_path):
    '''
    Description
    ------------
    A Leica proprietary image reader custom-made by Ernest to extract a multidimentional array from a .lif file.
    
    Parameters
    ------------
    image_path : string
                The path of the image.
    
    Returns
    ------------
    image_stack : array
                
    
    fields_of_view : list
                    Indeces corresponding to the fields of view.
    
    image_name : string
                The filename.
    '''
    
    print('Importing image...')
    lif_image = LifFile(image_path)
    image_path = lif_image.filename
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    img_list = [i for i in lif_image.get_iter_image()]
    
    all_images_names = []
    
    num = 0
    
    fields_of_view = range(len(img_list))
    
    for roi in img_list:
        number_channels = len([i for i in roi.get_iter_c(t=0, z=0)])
        first_img = np.uint16(np.array(roi.get_frame(z=0, t=0, c=0)))
        image = np.empty((first_img.shape[0], first_img.shape[1], number_channels))
    
        
        if num == 0:
            image_stack = np.empty((len(img_list), first_img.shape[0], first_img.shape[1],number_channels))
        
        for ch in range(number_channels):
            image[:,:, ch] = np.uint16(np.array(roi.get_frame(z=0, t=0, c=ch))) # plus 1 to avoid 0s
        
        image_stack[num,:,:,:] = image
        all_images_names.append(roi.name)
        num += 1
    
    print('Image imported!')
    
    return image_stack, fields_of_view, image_name

def nd2_image_reader(image_path):
    '''
    Description
    ------------
    A Nikon proprietary image reader custom-made by Ernest to extract a multidimentional array from a .nd2 file.
    
    Parameters
    ------------
    image_path : string
                The path of the image.
    
    Returns
    ------------
    image_stack : array
                
    
    fields_of_view : list
                    Indeces corresponding to the fields of view.
    
    image_name : string
                The filename.
                
    image.metadata : dictionary
                    The metadata.
    '''
    
    print('Importing image...')
    
    image = ND2Reader(image_path)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    # image.metadata['fields_of_view'] = [0]
    # image_stack slice order: fields of view, time frames, y axis, x axis, colors, z
    if image.metadata['z_levels']:#==True: # if there's a z stack
        image_stack = np.empty((len(image.metadata['fields_of_view']), len(image.metadata['frames']),
                                image.metadata['height'], image.metadata['width'], 
                                len(image.metadata['channels']),image.metadata['z_levels'][-1]+1),dtype=np.uint16)
        for fov in image.metadata['fields_of_view']:
            for ch in range(len(image.metadata['channels'])):
                for frame in range(len(image.metadata['frames'])):
                    for z in range(len(image.metadata['z_levels'])):
                         current_frame = ND2Reader.get_frame_2D(image, c=ch, t=frame, v=fov, z=z)
                         image_stack[fov, frame,:,:,ch,z] = current_frame
    else:
        image_stack = np.empty((len(image.metadata['fields_of_view']), len(image.metadata['frames']),
                                image.metadata['height'], image.metadata['width'], 
                                len(image.metadata['channels'])), dtype=np.uint16)

        for fov in image.metadata['fields_of_view']:
            for ch in range(len(image.metadata['channels'])):
                for frame in range(len(image.metadata['frames'])):
                     current_frame = ND2Reader.get_frame_2D(image, c=ch, t=frame, v=fov)
                     image_stack[fov, frame,:,:,ch] = current_frame
    fields_of_view = image.metadata['fields_of_view']
    
    print('Image imported!')
    
    return image_stack, fields_of_view, image_name, image.metadata

def nd2_image_reader_new(image_path):
    '''
    Description
    ------------
    A Nikon proprietary image reader custom-made by Ernest to extract a multidimentional array from a .nd2 file.
    
    Parameters
    ------------
    image_path : string
                The path of the image.
    
    Returns
    ------------
    image_stack : array
                
    
    fields_of_view : list
                    Indeces corresponding to the fields of view.
    
    image_name : string
                The filename.
                
    image.metadata : dictionary
                    The metadata.
    '''
    
    print('Importing image...')
    
    image = nd2.ND2File(image_path)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    image_stack = np.asarray(image)
    # image_stack slice order: fields of view, time frames, y axis, x axis, colors

    

    fields_of_view = 1
    if 'P' in image.sizes:
        fields_of_view = image.sizes['P']
    
    
    print('Image imported!')
    
    return image_stack, fields_of_view, image_name, image.metadata

