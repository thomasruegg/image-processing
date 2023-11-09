#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 7 Template
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fft2, ifft2

# Parameter
on = 50  # Standard deviation of white noise

# Load image
f = plt.imread("../../images/dog.jpg")

# Tranform image to grayscale and float
if f.ndim > 2:
    f = np.mean(f, axis=2)
f = np.array(f, dtype=float)
(M, N) = f.shape


# TODO:
# Variables
# gb: shifted and noisy image
# gc: shifted back image (still noisy)
# r: normalized cross-correlation matrix of f and gb
# %% Shift image 2
# Hint: See np.roll(): Shifts images circularly
g = np.zeros_like(f)
# TODO Shift image f -> g

# add white noise (to demonstrate robustness of method)
gb = g + np.std(f) * np.random.randn(M, N)

# %% Phase correlation
# Tranform both images

# Calculate r
r = np.zeros_like(g)
# TODO calculate phase correlation matrix -> r

# Find peak in r
# Hint: see np.unravel_index() and np.argmax()

# Shift back image
gc = np.zeros_like(g)
# TODO shift back g -> gc

# %% Display results
fig, ax = plt.subplots(
    2, 2, num=1, clear=True, constrained_layout=True, figsize=(10, 15)
)
ax[0][0].imshow(f)
ax[0][0].title.set_text("Original")
ax[0][1].imshow(gb)
ax[0][1].title.set_text("Shifted & Noisy")
ax[1][0].imshow(gc)
ax[1][0].title.set_text("Shifted back")
ax[1][1].imshow(r)
ax[1][1].title.set_text("Crosscorrelation")
for a in ax.flatten():
    a.axis("off")

Y = np.arange(0, M)
X = np.arange(0, N)
X, Y = np.meshgrid(X, Y)
fig2 = plt.figure(figsize=(15, 15), constrained_layout=True)
ax2 = fig2.gca(projection="3d")
r2 = r - r.min()
ax2.plot_surface(X, Y, r2, cmap=cm.viridis)
plt.show()
