#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 6 Template
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft

# Generate time series
n_f = np.random.randint(50, 80)
n_h = np.random.randint(30, 60)
n = max(n_f, n_h)
f = np.sin(4 * np.pi * np.arange(n_f) / n_f)
h = np.cos(6 * np.pi * np.arange(n_h) / n_f)

# %% Linear convolution

# your implementation
fh = np.zeros(f.shape[0] + h.shape[0] - 1)
# TODO calculate linear convolution of f and h -> fh

# use numpy function
fh2 = np.convolve(f, h)

# %% Cyclic convolution

# use FFT
fh_z = np.real(ifft(fft(f, n) * fft(h, n)))

# in time domain (optional)
# TODO calculate linear convolution of f and h -> fh_z2

# %% Display results
fig, ax = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
ax[0].plot(f)
ax[0].plot(h)
ax[0].legend(["f", "h"])
ax[1].plot(fh2)
ax[1].plot(fh_z)
ax[1].legend(["Linear Conv", "Circular"])
for a in ax:
    a.axis("on")
plt.show()
