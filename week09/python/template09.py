#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 8 Template
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2

# Load image
img = plt.imread("../../images/dog.jpg")
f = img.copy()

# Tranform image to grayscale and float
if f.ndim > 2:
    f = np.mean(f, axis=2)
f = np.array(f, dtype=float)
(M, N) = f.shape


# Variables
# g1: filtered image in spatial domain
# g2: filtered image in frequency domain with filter designed in frequency domain
# g3: filtered image in frequency domain with filter designed in spatial domain
# H:  filter in frequency domain

# %% Filtering in spatial domain
g1 = np.zeros_like(f)
# TODO Filter f in spatial domain -> g1

# %% Filtering in frequency domain
P = 2 * M
Q = 2 * N

# generate spatial filter
h = np.zeros(9)
h[1::2] = 1
h = np.reshape(h, (3, 3))

# FFT
# TODO calculate FFT of f
# TODO calculate FFT of h

# Generate frequency filter
H = np.zeros((P, Q))
# TODO Generate filter in frequency domain -> H

# Filtering
g2 = np.zeros_like(f)
g3 = np.zeros_like(f)
# TODO ifft(F * H) -> g2
# TODO ifft(F * fft(h)) -> g3

# %% Display results
fig, ax = plt.subplots(
    2, 2, num=1, clear=True, constrained_layout=True, figsize=(15, 15)
)
ax[0, 0].imshow(f, cmap="gray")
ax[0, 0].title.set_text("Original")
ax[0, 1].imshow(g1, cmap="gray")
ax[0, 1].title.set_text("Filtered in Spatial Domain")
ax[1, 0].imshow(g2, cmap="gray")
ax[1, 0].title.set_text("Filtered in Frequency Domain, Frequency Design")
ax[1, 1].imshow(g3, cmap="gray")
ax[1, 1].title.set_text("Filtered in Frequency Domain, Spatial Design")
for a in ax.flatten():
    a.axis("off")

fig2, ax2 = plt.subplots(num=2, clear=True, constrained_layout=True)
p = ax2.imshow(np.fft.fftshift(H), cmap="viridis")
ax2.title.set_text("Filter in Frequency Domain")
fig2.colorbar(p)
plt.show()
