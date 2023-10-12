#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Python Excercise Lab 4

Histogramm implementation

@author: Patrik Müller
@date:   10.09.2019
"""

import matplotlib.pyplot as plt
import numpy as np


# Histogram function
def histogram(img, levels):
    h = np.zeros(np.size(levels))
    for level in levels:
        h[level] = np.sum(img == level)
    h = h / np.sum(h)
    return h


if __name__ == "__main__":
    # Number of grayscale values
    nbins = 256
    graylevels = range(nbins)

    # Load image
    f = plt.imread("../../images/lions.jpg")

    # Tranform image to grayscale and float
    if f.ndim > 2:
        f = np.mean(f, axis=2)
    f = np.array(f, dtype=float)

    # Cumulated histogram -> transformation function
    h = histogram(f, graylevels)
    T = (nbins - 1) * np.cumsum(h)
    T = np.array(np.round(T), dtype="uint8")

    # Transform image
    f = np.array(f, dtype="uint8")
    g = T[f]

    # Calculate histogram of transformed image
    p = histogram(g, graylevels)

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
    ax[0, 2].plot(graylevels, np.cumsum(h))
    ax[1, 0].imshow(g, cmap="gray")
    ax[1, 1].plot(graylevels, p)
    ax[1, 2].plot(graylevels, np.cumsum(p))
    for a, title in zip(
        ax[:, 1:].flatten(),
        2 * ["Normalized Histogram", "Cummulative Probability Density"],
    ):
        a.grid(b=True)
        ylim = 1.05 * np.max(a.lines[0].get_ydata())
        a.axis([0, nbins - 1, 0, ylim])
        a.set_title(title)
    for a, title in zip(ax[:, 0], ["Original Image", "Histrogram Equalized Image"]):
        a.axis("off")
        a.set_title(title)
    plt.show()
