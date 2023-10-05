#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 3 Template
"""

import matplotlib.pyplot as plt
import numpy as np

# Parameter
N = 2 ** 8
graylevels = range(N)

# Load image
img = plt.imread("../images/lions.jpg")

# Transform image to grayscale
f = img.copy()
if f.ndim > 2:
    f = np.mean(f, axis=2)

f = f.astype(np.uint8)

# Calculate normalized histogram
h = np.zeros_like(graylevels, dtype=float)
for graylevel in f.flatten():
    h[graylevel] += 1


# Calculate normalized histogram
# hist, bin_edges = np.histogram(f.ravel(), bins=N, range=(0, N))
# h = hist / hist.sum()  # Normalize


# Calculate cumulative probability density function (See np.cumsum)
c = np.zeros_like(graylevels)
c = np.cumsum(h)

# Display result
fig, ax = plt.subplots(
    2, 2, sharex="col", num=1, clear=True, figsize=(15, 10), constrained_layout=True
)
ax = ax.T.flatten()
ax[0].imshow(img, cmap="gray")
ax[0].title.set_text("Original Image")
ax[0].axis("off")
ax[1].imshow(f, cmap="gray")
ax[1].title.set_text("Grayscale Image")
ax[1].axis("off")
ax[2].plot(graylevels, h)
ax[2].title.set_text("Histogram (Probability Density Function)")
ax[3].plot(graylevels, c)
ax[3].title.set_text("Cumulative Probability Density Function")
for a in ax[2:]:
    a.grid()
    ylim = np.max(a.lines[0].get_ydata())
    a.axis([0, N - 1, 0, 1.05 * ylim])
plt.show()
