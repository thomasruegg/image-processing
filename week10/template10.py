#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 9 Template
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2

# TODO Set the parameters to the right values
# Parameter
gamma_H = 1
gamma_L = 0
c = 1
D0_squ = 1 ** 2

# Load image
img = plt.imread("../../images/lions.jpg")

# Tranform image to grayscale and float
if img.ndim > 2:
    img = np.mean(img, axis=2)
img = np.array(img, dtype=float)
f = img.copy()
(M, N) = f.shape

# Variables
# fil:   Illuminated image
# ffilt: Filtered image with holomorphic filter
# H:     Holomorphic filter

#%% Illuminate
# illumin = np.sin(np.matmul(np.expand_dims(np.arange(1,M + 1), axis=1),
#                    np.transpose(np.expand_dims(np.arange(1,N + 1), axis=1)))\
#                    /(M*N)*np.pi/4) + 0.1
# illumin = np.matmul(np.expand_dims(np.arange(1,M + 1), axis=1),
#                    np.transpose(np.expand_dims(np.arange(1,N + 1), axis=1)))\
#                    /(M*N)*0.9 + 0.1
# illumin = np.matmul(np.expand_dims(np.arange(1,M + 1)**2, axis=1),
#                    np.transpose(np.expand_dims(np.arange(1,N + 1)**2, axis=1)))\
#                    /((M*N)**2)*0.9 + 0.05
illumin = (
    np.matmul(
        np.ones((M, 1)), np.transpose(np.expand_dims(np.arange(1, N + 1) ** 2, axis=1))
    )
    / N ** 2
    * 0.9
    + 0.05
)
# TODO: Illuminate f

#%% Logarithmitizing

#%% FFT

#%% Generate filter in frequency domain

#%% Filter image

#%% Back transformation

#%% Display results
fig, ax = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
ax[0].imshow(img, cmap="gray")
ax[0].title.set_text("Original")
ax[1].imshow(fil, cmap="gray")
ax[1].title.set_text("Illuminated")
ax[2].imshow(ffilt, cmap="gray")
ax[2].title.set_text("Filtered")
for a in ax:
    a.axis("off")

fig2, ax2 = plt.subplots(figsize=(10, 10), constrained_layout=True)
p = ax2.imshow(np.roll(H, [M, N], axis=(0, 1)))
ax2.title.set_text("Shifted Filter")
fig2.colorbar(p)
plt.show()
