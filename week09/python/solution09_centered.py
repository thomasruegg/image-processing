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

# Generate the impulse response of the spatial filter
h = np.zeros(9)
h[1::2] = 1
h = np.reshape(h, (3, 3))

# FFT
F = fft2(f, [P, Q])
H3 = fft2(h, [P, Q])
# Centered FFT
x = np.reshape(np.arange(P), (P, 1))
y = np.reshape(np.arange(Q), (1, Q))
fp = np.pad(f, ((0, P-M), (0, Q-N)))
fc = fp*pow(-1, x+y)
Fc = fft2(fc)

# Discrete frequency response of the filter by sampling the Fourier Transform
u = np.reshape(np.arange(P), (P, 1)) # vertical frequency indices
v = np.reshape(np.arange(Q), (1, Q)) # horizontal frequency indices
H2 = 0.5 * (np.cos(2 * np.pi * v / Q) + np.cos(2 * np.pi * u / P))
# Centered discrete frequency response
Hc2 = 0.5 * (np.cos(2 * np.pi * (v-Q/2) / Q) + np.cos(2 * np.pi * (u-P/2) / P))

# Filtering
G2 = F * H2
G3 = F * H3
G4 = Fc * Hc2
# Back transformation

g2 = np.real(ifft2(G2)[:M, :N])
g3 = np.real(ifft2(G3)[:M, :N])
g4 = np.real(ifft2(G4)*pow(-1, x+y))[:M, :N]
# %% Display results
fig, ax = plt.subplots(
    2, 3, num=1, clear=True, constrained_layout=True, figsize=(8, 8)
)
ax[0, 0].imshow(f, cmap="gray")
ax[0, 0].title.set_text("Original")
ax[0, 1].imshow(g1, cmap="gray")
ax[0, 1].title.set_text("Filtered in Spatial\n Domain")
ax[0, 2].imshow(g2, cmap="gray")
ax[0, 2].title.set_text("Filtered in Frequency\n Domain, Frequency Design")
ax[1, 1].imshow(g3, cmap="gray")
ax[1, 1].title.set_text("Filtered in Frequency\n Domain, Spatial Design")
ax[1, 2].imshow(g4, cmap="gray")
ax[1, 2].title.set_text("Filtered in Frequency\n Domain, Centered Frequency Design")

for a in ax.flatten():
    a.axis("off")

fig2, ax2 = plt.subplots(2,2, num=2, clear=True, constrained_layout=True, figsize=(8,8))
p = ax2[0, 0].imshow(H2, cmap="viridis")
ax2[0, 0].title.set_text("H2: Filter\n in Frequency Domain")
fig2.colorbar(p)
ax2[0, 1].imshow(Hc2,cmap="viridis")
ax2[0, 1].title.set_text("Hc2: Centered Filter\n in Frequency Domain")
ax2[1, 0].imshow(np.fft.fftshift(H2), cmap="viridis")
ax2[1, 0].title.set_text("H2: centered with\n handy fftshift() method")
plt.show()
