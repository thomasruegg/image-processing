#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 12 Template
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2

# Parameter
n_std = 10  # noise standard deviation
K = n_std / 2  # wiener
GA = 1e-10  # constrained least square

# Load image
img = plt.imread("../../images/dog.jpg")

# Tranform image to grayscale and float
if img.ndim > 2:
    img = np.mean(img, axis=2)
img = np.array(img, dtype=float)
(M, N) = img.shape
f = img.copy()

# Variables
# g:   Blurred and noisy image
# fhw: Image filtered with Wiener filter
# fh:  Image filtered with Constrained Least Squares

#%% Bluring Filter H, equivalten to motion-blur with 4 images (see exercise 11)

#%% Blur image

# filtering

#%% Add noise

#%% Restoration filtering

#%% Display Results
fig, ax = plt.subplots(2, 2, figsize=(15, 15), constrained_layout=True)
ax[0, 0].imshow(img, cmap="gray")
ax[0, 0].title.set_text("Original")
ax[0, 1].imshow(g, cmap="gray")
ax[0, 1].title.set_text("Blurred and Noisy")
ax[1, 0].imshow(fh, cmap="gray")
ax[1, 0].title.set_text("Wiener")


for a in np.reshape(ax, 4):
    a.axis("off")
plt.show()
