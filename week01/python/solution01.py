#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Python Excercise Lab 1

Show images with Python

@author: Patrik Müller
@date:   10.09.2019
"""

import matplotlib.pyplot as plt

# Load images
img = plt.imread("../../images/lions.jpg")
img2 = plt.imread("../../images/dog.jpg")

# %% Examples
# All the examples below work also with double values

# Example 1: Display a RGB image
fig1, ax1 = plt.subplots()
ax1.imshow(img)

# Example 2: Display a grayscale image without colormap specified
fig2, ax2 = plt.subplots()
ax2.imshow(img2)

# Example 3: Set value range
fig3, ax3 = plt.subplots()
p = ax3.imshow(img2, vmin=50, vmax=150)

# Example 4: Display a grayscale image with colormap
fig4, ax4 = plt.subplots()
p = ax4.imshow(img2, cmap="gray")
plt.show()
