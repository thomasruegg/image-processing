#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Python Excercise Lab 5

Image Filtering

@author: Patrik Müller
@date:   11.09.2019
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import gc
import cv2

# Parameters
k = 1.2  # highboost coefficient
m = 10  # neighborhood radius
n = 2 * m + 1  # filter size

# Load image
f = plt.imread("../images/lions.jpg")

# Tranform image to grayscale and float
if f.ndim > 2:
    f = np.mean(f, axis=2)
f = np.array(f, dtype=float)

# %% image filter
# add black border to image

gc.disable()

def boxfilter_implemented_python_only(f, m):
    (h, w) = np.shape(f)
    ft = np.zeros((h + 2 * m, w + 2 * m))
    ft[m:-m, m:-m] = f
    ft = np.array(ft, dtype="uint16")
    # low pass filter
    (h2, w2) = np.shape(ft)
    g = np.zeros_like(f, dtype="float")
    r = np.arange(m, h2 - m)
    c = np.arange(m, w2 - m)
    for rm in range(-m, m + 1):
        for cm in range(-m, m + 1):
            g += ft[r + rm, :][:, c + cm]
    return g


def boxfilter_by_copying_image_blockwise(f, m):
    (h, w) = np.shape(f)
    ft = np.zeros((h + 2 * m, w + 2 * m))
    ft[m:-m, m:-m] = f
    ft = np.array(ft, dtype="uint16")
    # low pass filter
    (h2, w2) = np.shape(ft)
    g = np.zeros_like(f, dtype="float")
    r = np.arange(m, h2 - m)
    c = np.arange(m, w2 - m)
    for rm in range(-m, m + 1):
        for cm in range(-m, m + 1):
            g += ft[r + rm, :][:, c + cm]
    return g


def boxfilter_by_copying_image_row_and_columnwise(f, m):
    (h, w) = np.shape(f)
    ft = np.zeros((h + 2 * m, w + 2 * m))
    ft[m:-m, m:-m] = f
    ft = np.array(ft, dtype="uint16")
    # low pass filter
    (h2, w2) = np.shape(ft)
    g = np.zeros((h, w + 2 * m), dtype="float")
    p = np.zeros((h, w), dtype="float")
    r = np.arange(m, h2 - m)
    c = np.arange(m, w2 - m)
    for rm in range(-m, m + 1): # column wise convolution
        g += ft[r + rm, :]
    for cm in range(-m, m + 1): # row wise convolution
        p += g[:, c + cm]
    return p 




start1 = time.perf_counter_ns()
g = boxfilter_by_copying_image_blockwise(f, m)
stop1 = time.perf_counter_ns()
print(f'blockwise: runtime = {(stop1-start1)/1e6} ms')

start2 = time.perf_counter_ns()
h = boxfilter_by_copying_image_row_and_columnwise(f, m)
stop2 = time.perf_counter_ns()
print(f'separable: runtime = {(stop2-start2)/1.e6} ms')

start3 = time.perf_counter_ns()
i = cv2.boxFilter(f, ddepth=-1, ksize=(2*m+1, 2*m+1), borderType=cv2.BORDER_ISOLATED)
stop3 = time.perf_counter_ns()
print(f'openCV: runtime = {(stop3-start3)/1e6} ms')


plt.imshow(g, cmap='gray')
plt.show()
plt.imshow(h, cmap ='gray')
plt.show()
plt.imshow(i, cmap ='gray')
plt.show()



