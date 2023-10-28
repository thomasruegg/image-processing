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

# Parameters
k = 1.2  # Highboost coefficient
m = 10  # Neighborhood radius
n = 2 * m + 1  # Filter size

# Load image
# f = plt.imread("modules/ipcv-1/images/lions.jpg")
f = plt.imread("../images/lions.jpg")

# Tranform image to grayscale and float
if f.ndim > 2:
    f = np.mean(f, axis=2)
f = np.array(f, dtype=float)

# %% image filter
# Zero padding: add border of zeros with required width to image
(h, w) = np.shape(f)
ft = np.zeros((h + 2 * m, w + 2 * m))
ft[m:-m, m:-m] = f  # Notice that indexing in python is cyclic!
ft = np.array(ft, dtype="uint16")

# Define the smooting filter kernel
(h2, w2) = np.shape(ft)

# Variante 1 of convolving ft with the box kernel of dimension n x n with n = 2 * m + 1
g = np.zeros_like(f, dtype="float")
r = np.arange(m, h2 - m)
c = np.arange(m, w2 - m)
for rm in range(-m, m + 1):
    for cm in range(-m, m + 1):
        g += ft[r + rm, :][:, c + cm]

# Variante 2 of convolving ft with the box kernel of dimension n x n with n = 2 * m + 1
# g = np.zeros_like(ft, dtype='float')
# for rm in range(0, 2 * m + 1):
#    for cm in range(0, 2 * m + 1):
#        g[rm : h + rm, cm : w + cm] += f
# g = g[m:-m,m:-m]

g = g / (n * n)  # Normalize the magnitude to keep image brightness unchanged after smooting
g = np.array(g, dtype="uint8")  # Convert floating point image back to uint8

# %% Unsharp masking
sharp = g + k * (f - g)  # Apply highboost, here f-g is the high-pass filtered image.

# %% calculate histograms, just to see that we get the additional contrast without much change in histogram shape
sharp = np.array(sharp, dtype="uint8")

x = range(256)
bins = np.arange(257) - 0.5
h1, _ = np.histogram(f, bins, density=True)
h2, _ = np.histogram(g, bins, density=True)
h3, _ = np.histogram(sharp, bins, density=True)
# %% Display results
fig, ax = plt.subplots(
    3,
    3,
    sharex="row",
    sharey="row",
    figsize=(15, 15),
    num=1,
    clear=True,
    constrained_layout=True,
    # gridspec_kw={'height_ratios': [2, 1]}
)
for a in ax[1:, :].flatten():
    a.grid(True)
ax = ax.T.flatten()
ax[0].imshow(f, cmap="gray")
ax[0].title.set_text("Original Image")
ax[1].plot(x, h1)
ax[2].plot(x, np.cumsum(h1))
ax[3].imshow(g, cmap="gray")
ax[3].title.set_text("Unsharp Image")
ax[4].plot(x, h2)
ax[5].plot(x, np.cumsum(h2))
ax[6].imshow(sharp, cmap="gray")
ax[6].title.set_text("Sharpened Image")
ax[7].plot(x, h3)
ax[8].plot(x, np.cumsum(h3))
for a in ax[::3]:
    a.axis("off")
    a.grid(False)
plt.show()
