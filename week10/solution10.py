#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Python Excercise Lab 9

Homomorphic Filter Implementation

@author: Patrik Müller
@date:   11.09.2019
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2

# Parameter
gamma_H = 1
gamma_L = 0
c = 1
D0_squ = 5 ** 2

# Load image
img = plt.imread("images/lions.jpg")

# Tranform image to grayscale and float
if img.ndim > 2:
    img = np.mean(img, axis=2)
img = np.array(img, dtype=float)
f = img.copy()
(M, N) = f.shape

#%% Illuminate
# illumin = np.sin(np.matmul(np.expand_dims(np.arange(1,M + 1), axis=1),
#                    np.transpose(np.expand_dims(np.arange(1,N + 1), axis=1)))\
#                    /(M*N)*np.pi/4) + 0.1
# illumin = np.matmul(np.expand_dims(np.arange(1,M + 1), axis=1),
#                    np.transpose(np.expand_dims(np.arange(1,N + 1), axis=1)))\
#                    /(M*N)*0.9 + 0.1
# illumin = np.matmul(np.expand_dims(np.arange(1,M + 1)**2, axis=1),
#                    np.transpose(np.expand_dims(np.arange(1,N + 1)**2, axis=1)))\
#                    /((M*N)**2)*0.9 + 0.05
illumin = (
    np.matmul(
        np.ones((M, 1)), np.transpose(np.expand_dims(np.arange(1, N + 1) ** 2, axis=1))
    )
    / N ** 2
    * 0.9
    + 0.05
)

f = f * illumin

#%% Logarithmitizing
flog = np.log(f + 1)

#%% FFT
P = 2 * M
Q = 2 * N
x = np.hstack((flog, np.fliplr(flog)))
Flog = fft2(np.vstack((x, np.flipud(x))))

#%% Generate filter in frequency domain
u = np.reshape(np.arange(P), (P, 1))
v = np.reshape(np.arange(Q), (1, Q))
H = (gamma_H - gamma_L) * (
    1 - np.exp(-c * ((u - P / 2) ** 2 + (v - Q / 2) ** 2) / D0_squ)
) + gamma_L

# Alternative: for loop (slow!)
# H = np.zeros((P,Q))
# for u in range(P):
#    for v in range(Q):
#        D_squ = (u-P/2)**2 + (v-Q/2)**2
#        H[u,v] = (gamma_H - gamma_L)*(1-np.exp(-c*D_squ/D0_squ))+ gamma_L

H = np.roll(H, [M, N], axis=(0, 1))
#%% Filter image
Ffilt = Flog * H

#%% Back transformation
ffilt = np.exp(np.real(ifft2(Ffilt))) - 1
ffilt = ffilt[:M, :N]

#%% Display results
fig, ax = plt.subplots(3, 1, figsize=(10, 15))
ax[0].imshow(img, cmap="gray")
ax[0].title.set_text("Original")
ax[1].imshow(f, cmap="gray")
ax[1].title.set_text("Illuminated")
ax[2].imshow(ffilt, cmap="gray")
ax[2].title.set_text("Filtered")
for a in ax:
    a.axis("off")

fig2, ax2 = plt.subplots(figsize=(10, 10))
p = ax2.imshow(np.roll(H, [M, N], axis=(0, 1)))
ax2.title.set_text("Shifted Filter")
fig2.colorbar(p)

plt.show()
