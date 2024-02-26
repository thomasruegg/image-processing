#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 13 Template
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

# Variables
# H:  Hue original image
# S:  Saturation original image
# I:  Intensity original image
# H2: Shifted hue
# S2: Shifted saturation
# I2: Shifted Intensity
# R2: Shifted red layer
# G2: Shifted green layer
# B2: Shifted blue layer
# f2: Shifted RGB image

#%% RGB to HSI
[R, G, B] = np.squeeze(np.split(f, 3, axis=2))

# Hue
# Important: add small falue -> no division by 0 !!

# Intensity

# Saturation

# manage bad case (R+G+B) = 0
S[(R + G + B) == 0] = 0

#%% Use rgb_to_hsv
f_hsi = rgb_to_hsv(f)
H_m = f_hsi[:, :, 0] * 360
S_m = f_hsi[:, :, 1]
I_m = f_hsi[:, :, 2]

#%% Manipulate H,S and/or I

#%% HSI to RGB

# RG-sector 0...120:
# important: convert angles to radiant!

# GB-sector 120...240

# BR-sector 240...360


#%% Display Results
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
