import os
import csv
import numpy as np
import pandas as pd



BASE_DIR = '/home/workspace/CarND-Behavioral-Cloning-P3/data'
FILE_NAME = 'driving_log.csv'
CORRECTION = 0.2

if __name__ == '__main__':
    data = pd.read_csv(os.path.join(BASE_DIR, FILE_NAME))
    data['steering_left'] = data['steering'] + CORRECTION
    data['steering_right'] = data['steering'] - CORRECTION
    
    center_data = data[['center', 'steering']]
    left_data = data[['left', 'steering_left']]
    right_data = data[['right', 'steering_right']]
    
    center_data = center_data.rename(index=str,
                                    columns={'center':'image'})
    left_data = left_data.rename(index=str, 
                                columns={'left':'image',
                                        'steering_left': 'steering'})
    right_data = right_data.rename(index=str, columns={'right':'image',
                                                      'steering_right':'steering'})
    
    data_frames = [center_data, left_data, right_data]
    final_data = pd.concat(data_frames)
   # save
    final_data.to_csv(os.path.join(BASE_DIR, 'driving.csv'), encoding='utf-8',             index=False)
    
    
