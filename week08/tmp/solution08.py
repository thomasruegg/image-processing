#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Python Excercise Lab 7

Phase Correlation

@author: Patrik Müller
@date:   11.09.2019
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fft2, ifft2
import os

# Parameter
on = 200  # Standard deviation of white noise

# Load image
print("cwd = " + os.getcwd())
f = plt.imread("../../images/dog.jpg")

# Tranform image to grayscale and float
if f.ndim > 2:
    f = np.mean(f, axis=2)
f = np.array(f, dtype=float)
(M, N) = f.shape

# %% Shift image 2
dr = np.random.randint(100, M - 100)
dc = np.random.randint(100, N - 100)
ga = np.roll(f, [dr, dc], axis=(0, 1))

# add white noise (to demonstrate robustness of method)
gb = ga + np.std(f) * np.random.randn(M, N)

# %% Phase correlation
# Tranform both images
Ga = fft2(f)
Gb = fft2(gb)

R = Ga * np.conj(Gb) / np.abs(Ga * np.conj(Gb))

# transform back
r = np.real(ifft2(R))

# Find peak
idx = np.unravel_index(np.argmax(r), r.shape)

# Shift back image
gc = np.roll(ga, idx, axis=(0, 1))

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
# r2 = np.exp(r-r.min())
# r2 = r2/r2.max()
ax[1][1].imshow(r)
ax[1][1].title.set_text("Crosscorrelation")
for a in ax.flatten():
    a.axis("off")

Y = np.arange(0, M)
X = np.arange(0, N)
X, Y = np.meshgrid(X, Y)
fig2 = plt.figure(figsize=(15, 15))
# ax2 = fig2.gca(projection="3d")
ax2 = fig2.add_subplot(111, projection='3d')
r2 = r - r.min()
ax2.plot_surface(X, Y, r2, cmap=cm.viridis)
plt.show()
plt.waitforbuttonpress()

