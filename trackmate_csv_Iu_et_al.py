#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 17:31:44 2022

@author: ernestiu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import math 
import os


path = '/track_mate.csv'

image_name = os.path.splitext(os.path.basename(path))[0]

data = pd.read_csv(path)

data = data.iloc[2: , :] # drop the first 2 rows

data.drop(index=data.index[0], axis=0, inplace=True) # drop the 2nd row


# print(data.head())

# convert strings from the csv file to numbers
data['POSITION_X'] = pd.to_numeric(data['POSITION_X']) 
data['POSITION_Y'] = pd.to_numeric(data['POSITION_Y']) 
data['FRAME'] = pd.to_numeric(data['FRAME'])
data['POSITION_T'] = pd.to_numeric(data['POSITION_T'])
data['TRACK_ID'] = pd.to_numeric(data['TRACK_ID'])


all_track_IDs = data['TRACK_ID'].unique() # find all the unique numbers in the column 'TRACK_ID'

all_data = pd.DataFrame()
all_track_analyses = pd.DataFrame()


cmap = plt.get_cmap('coolwarm')#'RdBu_r'
num_of_bins = 20 # this dictates the number of shades/bins in the lookup table
colors = [cmap(i) for i in np.linspace(0, 1, num_of_bins)]

line_color = 'blue' # or colors[color_code]
max_speed = 0.6 # in μm/min - this dictates the max speed allowed in the color map


track_limit = None # this controls the limit of the track length that will appear in the plot # 40 is good

graph_bound = 250 # # in μm; this controls the axis range

def find_nearest(array, value):
    '''
    This function finds the nearest value from an array that is closest to the value and returns its index.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


#### Plot settings begin ####
mpl.rcParams['axes.linewidth'] = 2 # set the value globally
fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
ax.set_aspect('equal')
ax = plt.gca()
ax.set_axisbelow(False) # set axes above the lines
# ax.spines['top'].set_color('none')
# ax.spines['left'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['bottom'].set_position('zero')
bound = max(abs(-graph_bound), abs(graph_bound))
plt.ylim(-bound, bound)     
plt.xlim(-bound, bound)

font_size = 17 

plt.yticks(fontname = 'Helvetica',fontsize=font_size)     
plt.xticks(fontname = 'Helvetica',fontsize=font_size)  
 
plt.ylabel('µm', loc='top', fontsize=font_size)
plt.xlabel('µm', loc='right', fontsize=font_size)


ax.tick_params(length = 10, width = 2) # tick size

#### Plot settings end ####

for track_id in all_track_IDs:
    
    if math.isnan(track_id): # if there's a nan
        print('Track {} was ignored'.format(str(track_id)))
    
    else:
        curret_track_data = data.loc[data['TRACK_ID'] == track_id]
        curret_track_data = curret_track_data.sort_values(by=['FRAME'], ascending=True)
         
        # print(curret_track_data.head())
         
        all_x = list(curret_track_data['POSITION_X'].reset_index(drop=True))
        all_y = list(curret_track_data['POSITION_Y'].reset_index(drop=True))
        time = list(curret_track_data['POSITION_T'].reset_index(drop=True))
        
        first_x = all_x[0]
        first_y = all_y[0]
        
         
        all_new_x = []     
        all_new_y = [] 
        all_segments = []
    
        for n, (x, y) in enumerate(zip(all_x, all_y)):
            all_new_x.append(x - first_x) # append the normalized x coordinate to a list
            all_new_y.append(y - first_y)
            
            if x is not all_x[-1]:
                # Calculate the Euclidean distance between 2 pts
                segment = math.dist([x, y] , [all_x[n+1], all_y[n+1]])
                all_segments.append(segment)
                
        all_new_x = pd.Series(all_new_x) # convert from a list to a Series
        all_new_y = pd.Series(all_new_y)
        
    
        current_data = pd.concat([all_new_x, all_new_y], names=['Position X', 'Position Y'], axis=1)
        all_data = pd.concat([all_data, current_data], axis=1)
        
        duration = (time[-1] - time[0])/60   # in minutes
        
        all_time_stamps = [round(time[pt] - time[-1]) for pt in range(len(time))] # compute all the time
        all_time_stamps.reverse()
        
        track_length = len(all_time_stamps)
        
        displacement = math.dist([first_x, first_y] , [all_x[-1], all_y[-1]]) # in μm
     
        avg_velocity = displacement / duration
        
        euclidean_dist = np.sum(all_segments) # in μm
        
        average_speed =  euclidean_dist / duration
        
        directionality = displacement / euclidean_dist
        
        ### new additions May 2024 (begin) ###
        
        average_dist_per_frame = np.average(all_segments) # in μm
        average_speed_per_frame = average_dist_per_frame / (time[1]/60) # um/min; time[1] because the second time stamp is the time interval
        

        
        cell_data = {'Cell ID': [track_id], 'Euclidean distance (μm)': [euclidean_dist], 'Displacement (μm)': [displacement], 'Time (s)': [duration],'Track length (frame)': [track_length],
                     'Avg Velocity (μm/min)': [avg_velocity], 'Avg Speed (μm/min)': [average_speed], 'Directionality': [directionality], 'Avg Speed per frame (μm/min)': [average_speed_per_frame]}
    
        
        cell_data = pd.DataFrame(cell_data)
        all_track_analyses = pd.concat([all_track_analyses, cell_data], ignore_index=True)
        
        # color_code finds the number that's closest to a number on the color scale defined in colors from the speed.
        color_code = find_nearest([np.linspace(0, max_speed, num_of_bins)], average_speed)
        # print(colors[color_code])
        
        # if no track_limit was set, create a plot without a limit
        if track_limit == None:
            ax.plot(all_new_x, all_new_y, alpha=0.8, linewidth=3, c = line_color, zorder=0)
            ax.scatter(all_new_x.iloc[-1], all_new_y.iloc[-1], c='black', zorder=1)
        
        # if track_limit was set, only tracks with displacement higher than the set limit will be plotted
        elif track_length >= track_limit:
            ax.plot(all_new_x[0:track_limit], all_new_y[0:track_limit], alpha=0.3, linewidth=3, c = line_color, zorder=0)#colors[color_code])
            ax.scatter(all_new_x[0:track_limit].iloc[-1], all_new_y[0:track_limit].iloc[-1], c='black', zorder=1)
    
         
all_track_analyses = all_track_analyses.set_index(all_track_analyses.columns[0])


plt.savefig(os.path.dirname(path) + '/' + image_name + '_tracks_img.pdf', bbox_inches='tight', pad_inches=0.5, transparent=True)#, transparent=True)
plt.show()


writer = pd.ExcelWriter(os.path.dirname(path) + '/' + image_name + '.xlsx', engine = 'xlsxwriter')
all_track_analyses.to_excel(writer, sheet_name = 'Tracks analyses')
all_data.to_excel(writer, sheet_name = 'XY coordinates')
writer.close()





