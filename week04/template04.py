#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 4 Template
"""

import matplotlib.pyplot as plt
import numpy as np


# Histogram function
def histogram(img, levels):
    # TODO calculate a normalized histogram of the image img
    pass


# Parameter
N = 2 ** 8
graylevels = range(N)

# Load image
img = plt.imread("../../images/lions.jpg")

# Tranform image to grayscale
f = img.copy()
if f.ndim > 2:
    f = np.mean(img, -1).astype("uint8")

# Cumulated histogram -> transformation function
h = np.zeros(N)
c = np.zeros_like(h)
# TODO calculate histogram of f -> h
# TODO calculate cummulative probability density of f -> c

# Transform image
g = np.zeros_like(f)
# TODO apply histogram equalization scheme to f -> g

# Calculate histogram of transformed image
ht = np.zeros_like(h)
ct = np.zeros_like(h)
# TODO calculate histogram of g -> ht
# TODO calculate cummulative probability density of g -> ct

# Display results
fig, ax = plt.subplots(
    2,
    3,
    num=1,
    clear=True,
    figsize=(15, 10),
    constrained_layout=True,
)
ax[0, 0].imshow(f, cmap="gray")
ax[0, 1].plot(graylevels, h)
ax[0, 2].plot(graylevels, c)
ax[1, 0].imshow(g, cmap="gray")
ax[1, 1].plot(graylevels, ht)
ax[1, 2].plot(graylevels, ct)
for a, title in zip(
    ax[:, 1:].flatten(), 2 * ["Normalized Histogram", "Cummulative Probability Density"]
):
    a.grid(b=True)
    ylim = 1.05 * np.max(a.lines[0].get_ydata())
    a.axis([0, N - 1, 0, ylim])
    a.set_title(title)
for a, title in zip(ax[:, 0], ["Original Image", "Histrogram Equalized Image"]):
    a.axis("off")
    a.set_title(title)
plt.show()
