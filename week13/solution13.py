#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Python Excercise Lab 13

RGB <-> HSI Conversion

@author: Patrik Müller
@date:   12.09.2019
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv

# Parameter

# Load image
img = plt.imread("../images/lions.jpg")
f = img / np.amax(img)

if f.ndim != 3:
    sys.exit("Color image needed. Stop Execution")

(M, N, D) = f.shape

# %% RGB to HSI
[R, G, B] = np.squeeze(np.split(f, 3, axis=2))

# Hue
# Important: add small falue -> no division by 0 !!
n = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-3
d = 2 * R - G - B
Phi = np.arccos(0.5 * d / n) / np.pi * 180
H = Phi
tmp = B > G
H[tmp] = 360 - Phi[tmp]

# Intensity
I = np.mean(f, 2)

# Saturation
Min = np.amin(f, axis=2)
S = (I - Min) / I

# manage bad case (R+G+B) = 0
S[(R + G + B) == 0] = 0

# %% Use rgb_to_hsv
f_hsi = rgb_to_hsv(f)
H_m = f_hsi[:, :, 0] * 360
S_m = f_hsi[:, :, 1]
I_m = f_hsi[:, :, 2]

# %% Shift Color
H2 = H.copy()
S2 = S.copy()
I2 = I.copy()
# H = np.mod(H + 90, 360)
# S = np.mod(S+0.4, 1);
# I = np.mod(I+0.2, 1);

# %% HSI to RGB
R2 = np.zeros((M, N))
G2 = R2.copy()
B2 = R2.copy()

H_old = H.copy()

# RG-sector 0...120:
m = (H_old >= 0) & (H_old < 120)
B2[m] = I[m] * (1 - S[m])
# important: convert angles to radiant!
R2[m] = I[m] * (
    1 + S[m] * np.cos(H[m] / 180 * np.pi) / np.cos((60 - H[m]) / 180 * np.pi)
)
G2[m] = 3 * I[m] - R2[m] - B2[m]

# GB-sector 120...240
H = H_old - 120
m = (H_old >= 120) & (H_old < 240)
R2[m] = I[m] * (1 - S[m])
G2[m] = I[m] * (
    1 + S[m] * np.cos(H[m] / 180 * np.pi) / np.cos((60 - H[m]) / 180 * np.pi)
)
B2[m] = 3 * I[m] - R2[m] - G2[m]

# BR-sector 240...360
H = H_old - 240
m = (H_old >= 240) & (H_old < 360)
G2[m] = I[m] * (1 - S[m])
B2[m] = I[m] * (
    1 + S[m] * np.cos(H[m] / 180 * np.pi) / np.cos((60 - H[m]) / 180 * np.pi)
)
R2[m] = 3 * I[m] - (G2[m] + B2[m])

H = H_old

f2 = np.stack([R2, G2, B2], axis=2)
f2 /= np.amax(f2)

# %% Display Results
fig, ax = plt.subplots(4, 3, figsize=(15, 14), constrained_layout=True)
ax[0, 0].imshow(R, cmap="Reds")
ax[0, 0].title.set_text("R Original")
ax[0, 1].imshow(G, cmap="Greens")
ax[0, 1].title.set_text("G Original")
ax[0, 2].imshow(B, cmap="Blues")
ax[0, 2].title.set_text("B Original")
ax[1, 0].imshow(H2, cmap="hsv")
ax[1, 0].title.set_text("H")
ax[1, 1].imshow(S2)
ax[1, 1].title.set_text("S")
ax[1, 2].imshow(I2)
ax[1, 2].title.set_text("I")
ax[2, 0].imshow(H, cmap="hsv")
ax[2, 0].title.set_text("H Shifted")
ax[2, 1].imshow(S)
ax[2, 1].title.set_text("S Shifted")
ax[2, 2].imshow(I)
ax[2, 2].title.set_text("I Shifted")
ax[3, 0].imshow(R2, cmap="Reds")
ax[3, 0].title.set_text("R Shifted")
ax[3, 1].imshow(G2, cmap="Greens")
ax[3, 1].title.set_text("G Shifted")
ax[3, 2].imshow(B2, cmap="Blues")
ax[3, 2].title.set_text("B Shifted")
for a in np.reshape(ax, ax.size):
    a.axis("off")

fig2, ax2 = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
ax2[0].imshow(f)
ax2[0].title.set_text("Original")
ax2[1].imshow(f2)
ax2[1].title.set_text("Shifted")

for a in np.reshape(ax2, ax2.size):
    a.axis("off")
plt.show()
