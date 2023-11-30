#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Python Excercise Lab 10

Adaptive Local Noise Reduction Filter & Median Filter

@author: Patrik Müller
@date:   11.09.2019
"""

import matplotlib.pyplot as plt
import numpy as np


def spNoise(image, prob):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    prob /= 2
    output = image.copy()
    (M, N) = image.shape
    n = np.random.rand(M, N)
    output[n < prob] = 0
    output[n > 1 - prob] = 255
    return output


def adaptiveMean(f, sigma_n, s=5):
    fpad = np.pad(f, s // 2, "constant")
    var_n = sigma_n ** 2
    loc_var = np.zeros_like(f)
    loc_mean = np.zeros_like(f)

    for ix in range(0, f.shape[1]):
        for iy in range(0, f.shape[0]):
            f_loc = fpad[iy : iy + s, ix : ix + s]
            loc_var[iy, ix] = np.var(f_loc)
            loc_mean[iy, ix] = np.mean(f_loc)

    loc_var[loc_var < var_n] = var_n

    fh = f - var_n / (loc_var + 1e-12) * (f - loc_mean)
    return fh


def adaptiveMedian(f, S_max, S):
    s_max = int(S_max / 2)
    (M, N) = f.shape
    fh = np.zeros((M, N))
    for ix in range(s_max, N - s_max + 1):
        for iy in range(s_max, M - s_max + 1):
            s = S // 2        
            while 1:
                f_loc = f[iy - s : iy + s, ix - s : ix + s]
                z_med = np.median(f_loc)
                z_min = np.amin(f_loc)
                z_max = np.amax(f_loc)
                if z_min < z_med and z_med < z_max:
                    # Level B
                    z_xy = f[iy, ix] 
                    if z_min < z_xy and z_xy < z_max:
                        fh[iy, ix] = z_xy
                    else:
                        fh[iy, ix] = z_med
                    break
                else:
                    if s == s_max:
                        fh[iy, ix] = z_med
                        break
                    s += 1
    return fh


if __name__ == "__main__":
    # Parameter
    sigma_n = 12  # Standard deviation of noise (normalized)
    p_sp = 0.2  # Salt or Pepper probability
    S_mean = 5  # size of local area
    S_median_max = 11  # Max size of local area
    S_median_start = 3  # Start size of local area

    # Load image
    f = plt.imread("../../images/dog.jpg")

    # Tranform image to grayscale and float
    if f.ndim > 2:
        f = np.mean(f, axis=2)
    f = np.array(f, dtype=float)
    (M, N) = f.shape

    #  Add noise
    f_gau = f + sigma_n * np.random.randn(M, N)
    f_sp = spNoise(f, p_sp)

    # Adaptive denoising filter
    f_filt1_gau = adaptiveMean(f_gau, sigma_n, S_mean)
    f_filt1_sp = adaptiveMean(f_sp, sigma_n, S_mean)
    f_filt2_gau = adaptiveMedian(f_gau, S_median_max, S_median_start)
    f_filt2_sp = adaptiveMedian(f_sp, S_median_max, S_median_start)

    # Display results
    fig2, ax2 = plt.subplots(constrained_layout=True)
    ax2.imshow(f, cmap="gray")
    ax2.title.set_text("Original")
    ax2.axis("off")

    fig, ax = plt.subplots(2, 3, figsize=(9, 13), constrained_layout=True)
    ax[0, 0].imshow(f_gau, cmap="gray")
    ax[0, 0].title.set_text("Gaussian Noise")
    ax[1, 0].imshow(f_sp, cmap="gray")
    ax[1, 0].title.set_text("Salt & Pepper Noise")
    ax[0, 1].imshow(f_filt1_gau, cmap="gray")
    ax[0, 1].title.set_text("Adaptive Noise Reduction (Gaussian Noise)")
    ax[1, 1].imshow(f_filt1_sp, cmap="gray")
    ax[1, 1].title.set_text("Adaptive Noise Reduction (S&P Noise)")
    ax[0, 2].imshow(f_filt2_gau, cmap="gray")
    ax[0, 2].title.set_text("Adaptive Median (Gaussian Noise)")
    ax[1, 2].imshow(f_filt2_sp, cmap="gray")
    ax[1, 2].title.set_text("Adaptive Median (S&P Noise)")
    #
    for a in np.reshape(ax, ax.size):
        a.axis("off")
    plt.show()
