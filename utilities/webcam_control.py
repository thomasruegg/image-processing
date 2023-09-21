#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:59:59 2020

@author: mup
"""

import cv2
import os
import numpy as np
from skimage.transform import downscale_local_mean

N_PARAMS = 42  # Number of parameters in settings file


class Webcam():
    """
        Class to read from a video stream (camera or video file)

        Args:
            port (int or string): Video port or file to open
            settings (dict or string)): Camera settings
            downscale (float): downscaling factor for streamed images
    """

    def __init__(self, port=0, settings=None, downscale=1):
        super(Webcam, self).__init__()

        self._stop = False
        self.scale = (downscale,)*2 + (1,)
        if settings is None:
            settings = {'frame_width': 2048, 'frame_height': 1536,
                        'exposure': -4, 'gain': 0}

        # Open the device at location 'port'
        print('Try to open camera...')
        self.cap = cv2.VideoCapture(port)

        # Check whether user selected video stream opened successfully.
        if not (self.cap.isOpened()):
            raise IOError("Could not open camera at port {}".format(port))
        print('Camera opened successfully')

        # Camera setting
        print('Write settings...')
        if isinstance(settings, str):
            # Read settings from file
            with open(settings, "r") as f:
                content = f.read()
                props = content.split(',')
                props = np.array(props, dtype='float')
                for i in range(N_PARAMS):
                    self.cap.set(i, props[i])
        else:
            # Read settings from dictionary
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['frame_width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['frame_height'])
            self.cap.set(cv2.CAP_PROP_GAIN, settings['gain'])
            self.cap.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'])
        print('Camera is initialized')

    def stop_camera(self, arg=None):
        """ Stop streaming

        Args:
            arg (object): Dummy parameter to use as event callback
        """
        self._stop = True

    def get_frame(self):
        """ Get next frame (image) from stream
        """
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27 or self._stop:
            return False, 0
        ret, frame = self.cap.read()
        frame = np.round(
            downscale_local_mean(frame, self.scale)).astype('uint8')
        return ret, frame

    def close(self,):
        """ Close connection to video stream
        """
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera = Webcam(0, downscale=1)
    print('Start streaming...')
    while 1:
        ret, frame = camera.get_frame()
        if not ret:
            break
        cv2.imshow('Image', frame)
    camera.close()
    print('Closed camera')
