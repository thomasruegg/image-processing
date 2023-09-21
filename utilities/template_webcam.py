#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:34:30 2020

@author: mup
"""

import cv2
import numpy as np
from webcam_control import Webcam

# Parameter
settings = {'frame_width': 2048, 'frame_height': 1536, 'exposure': -4,
            'gain': 0}  # manual camera settings
# settings_file = 'settings.txt'  # file name with settings
# (generated by 'get_settings.py')

downsampling_factor = 1  # downsampling factor (increases speed, but data is lost)

# Open video stream
camera = Webcam(port=0, settings=settings, downscale=downsampling_factor)

print('Stream from camera')
while(1):
    # %% Get frames
    # Capture frame-by-frame
    ret, frame = camera.get_frame()
    if not ret:
        break

    # %% Pre-processing
    f = np.mean(frame, -1)  # Convert to grayscale
    # f = np.rot90(f, k=2)  # rotate image by 180 degrees
    f = f.astype('uint8')  # Convert to uint8

    # %% Image processing

    # %% Update visualization
    # Display the resulting frame
    cv2.imshow("Press 'q' or 'Esc' to close window", f)

# When everything done, release the video stream
camera.close()
print('Closed camera')
