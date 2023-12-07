#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Python Excercise Lab 12

Motion Blur Reconstruction with Wiener filter and Constrained Least Squares

@author: Patrik Müller
@date:   12.09.2019
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2
from skimage.filters import window
import scipy

# Parameter
D = 10  # Width of the blurring effect
n_std = 1  # noise standard deviation
K = n_std / 200  # wiener... smaller K -> reduces blurring but increases noise amplification
# THE 200 SHOULD BE f_std or S_f (Power spectrum of f) BUT THAT WE CANNOT KNOW


# Load image
img = plt.imread("../images/dog.jpg")
# Tranform image to grayscale and float
if img.ndim > 2:
    img = np.mean(img, axis=2)
img = np.array(img, dtype=float)
(M, N) = img.shape
f = img.copy()

#%% Motion Blur modeled in the frequency domain
# Filter H, equivalent to motion-blur with D copies of shifted images along the diagonal with length D in units of pixels
# u = np.expand_dims(np.arange(M), -1)
# v = np.expand_dims(np.arange(N), 0)

# H = np.ones_like(v)
# for i in range(1, D + 1):
#     H = H + np.exp(-2j * i * np.pi * (1 * (u - M / 2) / M + (v - N / 2) / N))


#%% Motion Blur modeled in the spatial domain
h = np.zeros((M, N), dtype=float)
h[0:D,0:D] = np.eye(D)
h = h/D  # Normalize the sum to 1
H = fft2(h)  # Spectrum of blurring filter

#%% Blur image
F = fft2(f)  # Spectrum of image

# filtering
G = F * H  # Spectrum of blurred image
g = np.real(ifft2(G))  # blurred image without noise

#%% To make the blurred image more realistic remove the
# wrap arround effect of the cyclic convoluiton by windowing
a = 0.2  # a is proportional to the fade out width of the windows
wx = scipy.signal.windows.tukey(M, alpha = a)
wy = scipy.signal.windows.tukey(N, alpha = a)
w = wx.reshape((M,1)) @ wy.reshape((1,N))  # Window
gw = w*g  # Windowed blurred image

#%% Add noise
n = n_std * np.random.randn(M, N)
gwn = gw + n  # Windowed blurred image with noise

#%% Restoration filtering
G = fft2(gwn)  # Spectrum of blurred image with noise

Hw = np.conj(H) / (np.abs(H) ** 2 + K)
fhw = np.real(ifft2(Hw * G))


#%% Display Results
fig, ax = plt.subplots(2, 2, figsize=(15, 15), constrained_layout=True)
ax[0, 0].imshow(img, cmap="gray")
ax[0, 0].title.set_text("Original image")
ax[0, 1].imshow(g + n, cmap="gray")
ax[0, 1].title.set_text("Cyclic convolution with blurring kernel h")
ax[1, 0].imshow(gwn, cmap="gray")
ax[1, 0].title.set_text("Windowed blurred image plus noisy")
ax[1, 1].imshow(fhw, cmap="gray")
ax[1, 1].title.set_text("Reconstructed by Wiener filter")

for a in np.reshape(ax, 4):
    a.axis("off")
plt.show()
