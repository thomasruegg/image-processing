#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Python Excercise Lab 8

Filter Implementation in Frequency & Spatial Domain

@author: Patrik Müller
@date:   11.09.2019
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

# %% Filtering in spatial domain
dummy = np.pad(f, (1, 1), "constant")
g1 = dummy.copy()
for ix in range(1, M + 1):
    for iy in range(1, N + 1):
        tmp = dummy[ix, iy - 1]
        tmp += dummy[ix, iy + 1]
        tmp += dummy[ix - 1, iy]
        tmp += dummy[ix + 1, iy]
        g1[ix - 1, iy - 1] = tmp / 4
g1 = g1[1:-1, 1:-1]

# %% Filtering in frequency domain
P = 2 * M
Q = 2 * N

# generate spatial filter
h = np.zeros(9)
h[1::2] = 1
h = np.reshape(h, (3, 3))

# FFT
F = fft2(f, [P, Q])
H3 = fft2(h, [P, Q])

# Generate frequency filter
u = np.reshape(np.arange(P), (P, 1))
v = np.reshape(np.arange(Q), (1, Q))
H2 = 0.5 * (np.cos(2 * np.pi * v / Q) + np.cos(2 * np.pi * u / P))

# Filtering
G2 = F * H2
G3 = F * H3

# Back transformation
g2 = np.real(ifft2(G2)[:M, :N])
g3 = np.real(ifft2(G3)[:M, :N])

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
p = ax2.imshow(np.fft.fftshift(H2), cmap="viridis")
ax2.title.set_text("Filter in Frequency Domain")
fig2.colorbar(p)
plt.show()
