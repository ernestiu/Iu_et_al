#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:03:24 2023

@author: ernestiu
"""

import numpy as np
from skimage import io, filters, img_as_ubyte, measure
from skimage.morphology import disk, closing, white_tophat, remove_small_holes
import matplotlib.pyplot as plt
import os
from skimage.color import label2rgb
import sys
image_corr_module_path = '/Ratiometric images analysis package/'
sys.path.append(image_corr_module_path)
from image_corr_module import remove_spurious_pixels
from k_means_segmentation import k_means_seg
from skimage.filters import meijering


image_path = '/EM_image.tif'
k_means_cluster = 3

image_name = os.path.splitext(os.path.basename(image_path))[0]

img = io.imread(image_path)


# filter the image with median filter followed by a gaussian filter
img_smooth = filters.gaussian(img, sigma=0.3, preserve_range=True)



img_smooth = remove_spurious_pixels(img_smooth, threshold=int(np.mean(img_smooth)+np.std(img_smooth)))

#the white tophat is used to even out the dark and bright actin filaments
top_hat = white_tophat(img_smooth, footprint=disk(5))


img_smooth = img_smooth - top_hat


# use ridge detector to enhance actin
enhanced = meijering(img_smooth, sigmas=(1, 2, 3), black_ridges=False, mode="reflect")
img_smooth = enhanced*np.mean(img_smooth) + img_smooth



# mask = img_smooth > filters.threshold_minimum(img_smooth) + np.std(img_smooth)*0.2
kmeans, mask = k_means_seg(img_smooth, show_result=True, K=k_means_cluster) 



# perform a morphological closing on the mask

mask_close = closing(mask, footprint=disk(1))


mask_close = remove_small_holes(mask_close, area_threshold=20)
plt.imshow(mask_close)
plt.show()


labels = measure.label(mask_close, connectivity=2)  # label the actin mask image


props = measure.regionprops(labels) # get the region props


area_thres = 6000# set the area threshold; objects larger than this number will be included
    
filtered_mask = np.zeros(labels.shape, dtype=np.uint8)

for prop in range(len(props)):
    
    label = props[prop].label
    line_area = props[prop].area

    if line_area >= area_thres:

        filtered_mask[labels == label] = 255


image_label_overlay = label2rgb(~filtered_mask, image=img.astype(np.uint8), bg_label=0, image_alpha=1, alpha=0.3, colors=['red'])
plt.imshow(image_label_overlay)
plt.tight_layout()
plt.show()
    

io.imsave(os.path.dirname(image_path) + '/' + image_name + '_overlay.png', img_as_ubyte(image_label_overlay), check_contrast=False)


io.imsave(os.path.dirname(image_path) + '/' + image_name + '_mask.tif', img_as_ubyte(filtered_mask), check_contrast=False)

no_actin_pixels = np.sum(filtered_mask == 255)
actin_density = (no_actin_pixels/(labels.shape[0]*labels.shape[1])) * 100
print('Actin density (in %): {}'.format(round(actin_density, 2)))
print('Porosity (in %): {}'.format(round(100-actin_density, 2)))

labels = measure.label(~filtered_mask, connectivity=2)  # label the pore mask image

props = measure.regionprops(labels)

pore_area_list = []
for prop in range(len(props)):
    
    pore_area = props[prop].area
    pore_area_list.append(pore_area)

print('Average pore size (in nm^2): {}'.format(round(np.mean(pore_area_list), 2)))


