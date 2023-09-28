#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 2 Template
"""

import matplotlib.pyplot as plt
import numpy as np

# Parameter
N = 100  # Number of iterations

# Load image
img = plt.imread("../images/dog.jpg")

# If image is colored, transform to gray scale image
# note: -> Color images are stored in 3D arrays. In the 3rd dimesion are the
#          three different colors red, green, blue. Average along 3rd dimension.
if img.ndim > 2:
    img = np.mean(img, axis=2)
img = np.array(img, dtype=float)

# Expand in 3rd dimension and copy image N times
images = np.expand_dims(img, -1)
images = np.repeat(images, N, -1)

# Add noise to images
im_noisy = images + np.std(img) * np.random.randn(*images.shape)

# Calculate mean
# im_mean = np.zeros(img.shape)
# TODO Average over N noisy images and write result to im_mean
im_mean = np.mean(im_noisy, -1)

# Display the result
fig, ax = plt.subplots(
    1, 3, num=1, clear=True, figsize=(20, 20), constrained_layout=True
)
ax[0].imshow(img, cmap="gray")
ax[0].title.set_text("Original image")
ax[1].imshow(im_noisy[..., 0], cmap="gray")
ax[1].title.set_text("Noisy image")
ax[2].imshow(im_mean, cmap="gray")
ax[2].title.set_text("Denoised image")
for a in ax:
    a.axis("off")
plt.show()
