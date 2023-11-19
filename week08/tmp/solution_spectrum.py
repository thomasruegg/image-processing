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
from skimage.filters import window
import scipy

# Parameter
on = 200  # Standard deviation of white noise

# Load image
print("cwd = " + os.getcwd())
f = plt.imread("../../images/dog.jpg")

# Transform image to grayscale and float
if f.ndim > 2:
    f = np.mean(f, axis=2)
f = np.array(f, dtype=float)[:-1,:-1]

# Normalize the image data to be between 0 and 1
f = (f - f.min()) / (f.max() - f.min())

(M, N) = f.shape
print(f"shape = ({M}, {N})")
Mhalf = int(M/2)
Nhalf = int(N/2)

# Multiply image with fade-out window
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html
wx = scipy.signal.windows.tukey(M, alpha = 0.3)
wy = scipy.signal.windows.tukey(N, alpha = 0.3)
w = wx.reshape((M,1)) @ wy.reshape((1,N))
wf = w*f

# Compute spectrum and first then center the spectrum for displaying
F = np.fft.fft2(f)
WF = np.fft.fft2(wf)
W = np.fft.fft2(w)
# %% Display results
Fcentered_rolling = np.roll(F, (Mhalf, Nhalf), axis=(0, 1))
Wcentered_rolling = np.roll(W, (Mhalf, Nhalf), axis=(0, 1))
WFcentered_rolling = np.roll(WF, (Mhalf, Nhalf), axis=(0, 1))

# Premultiply the windowed image with so that the speactrum appears centered
x = np.arange(0, M).reshape((M, 1))
y = np.arange(0, N).reshape((1, N))
c = np.power(-1, x + y)
Fcentered_premultiplied = np.fft.fft2(f * c)
Wcentered_premultiplied = np.fft.fft2(w * c)
WFcentered_premultiplied = np.fft.fft2(wf * c)


def show_figures(f, w, wf, Fcentered, Wcentered, WFcentered):

    fig, ax = plt.subplots(
    2, 3, num=1, clear=True, constrained_layout=True, figsize=(10, 15)
    )
    ax[0][0].imshow(f, vmin=0, vmax=1)
    ax[0][0].title.set_text("image")
    ax[0][1].imshow(w, vmin=0, vmax=1)
    ax[0][1].title.set_text("window")
    ax[0][2].imshow(wf, vmin=0, vmax=1)
    ax[0][1].title.set_text("windowed image")
    ax[1][0].imshow(np.log(1+np.abs(Fcentered)))
    ax[1][0].title.set_text("Centered magnitue of spectrum")
    ax[1][1].imshow(np.log(1+np.abs(Wcentered)))
    ax[1][1].title.set_text("Centered magnitue of spectrum")
    ax[1][2].imshow(np.log(1+np.abs(WFcentered)))
    ax[1][2].title.set_text("Centered Magnitue of spectrum")
    fig.show()
    fig.waitforbuttonpress()


show_figures(f, w, wf, Fcentered_rolling, Wcentered_rolling, WFcentered_rolling)


def show_spectra(Fr, Wr, WFr, Fp, Wp, WFp):

    fig, ax = plt.subplots(
    2, 3, num=1, clear=True, constrained_layout=True, figsize=(10, 15)
    )
    ax[0][0].imshow(np.log(1+np.abs(Fr)))
    ax[0][0].title.set_text("centered by rolling")
    ax[0][1].imshow(np.log(1+np.abs(Wr)))
    ax[0][1].title.set_text("centered by rolling")
    ax[0][2].imshow(np.log(1+np.abs(WFr)))
    ax[0][1].title.set_text("centered by rolling")
    ax[1][0].imshow(np.log(1+np.abs(Fp)))
    ax[1][0].title.set_text("centered by premultiplying")
    ax[1][1].imshow(np.log(1+np.abs(Wp)))
    ax[1][1].title.set_text("centered by premultiplying")
    ax[1][2].imshow(np.log(1+np.abs(WFp)))
    ax[1][2].title.set_text("centered by premultiplying")
    fig.show()
    fig.waitforbuttonpress()

show_spectra(Fcentered_rolling, Wcentered_rolling, WFcentered_rolling, Fcentered_premultiplied, Wcentered_premultiplied, WFcentered_premultiplied)
