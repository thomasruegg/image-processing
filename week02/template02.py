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
img = plt.imread("../../images/dog.jpg")

# If image is colored, transform to gray scale image
# note: -> Color images are stored in 3D arrays. In the 3rd dimesion are the
#          three different colors red, green, blue. Average along 3rd dimension.


# Copy images N times


# Add noise to images
im_noisy = img + np.std(img) * np.random.randn(*img.shape)  # Noisy image example

# Calculate mean
im_mean = np.zeros(img.shape)
# TODO Average over N noisy images and write result to im_mean

# Display the result
fig, ax = plt.subplots(
    1, 3, num=1, clear=True, figsize=(20, 20), constrained_layout=True
)
ax[0].imshow(img, cmap="gray")
ax[0].title.set_text("Original image")
ax[1].imshow(im_noisy, cmap="gray")
ax[1].title.set_text("Noisy image")
ax[2].imshow(im_mean, cmap="gray")
ax[2].title.set_text("Denoised image")
for a in ax:
    a.axis("off")
plt.show()
