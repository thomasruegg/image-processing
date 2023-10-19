#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 5 Template
"""

import matplotlib.pyplot as plt
import numpy as np

# Parameters
k = 1.2  # highboost coefficient
m = 10  # neighborhood radius
n = 2 * m + 1  # filter size

# Load image
f = plt.imread("../../images/lions.jpg")

# Tranform image to grayscale and float
if f.ndim > 2:
    f = np.mean(f, axis=2)
f = np.array(f, dtype=float)

# %% image filter

# add black border to image

# low pass filter
g = np.zeros_like(f)
# TODO low pass filter f -> g

# %% Unsharp masking
sharp = np.zeros_like(f)
# TODO sharpen g with unsharp masking -> sharp

# %% Display results
fig, ax = plt.subplots(3, 1, figsize=(15, 15), constrained_layout=True)
ax[0].imshow(f, cmap="gray")
ax[0].title.set_text("Original Image")
ax[1].imshow(g, cmap="gray")
ax[1].title.set_text("Unsharp Image")
ax[2].imshow(sharp, cmap="gray")
ax[2].title.set_text("Sharpened Image")
for a in ax:
    a.axis("off")
plt.show()
