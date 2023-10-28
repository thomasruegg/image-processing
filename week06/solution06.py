#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Python Excercise Lab 6

Linear & Cyclic Convolution

@author: Martin Weisenhorn
@date:   27.10.2021
"""

import numpy as np
from numpy.fft import fft, ifft


N = 4  # Lenght of signal
M = 2  # Length of impulse response 
# Generate time series
s = np.zeros(N, dtype="int")
s[[0,1,2,3]] = 1
print(f'signal s = {s}')

h = np.zeros(M, dtype="int")
h[[0, 1]] = 1
print(f'filter h = {h}')

# Own implementation of the linear convolution
glc = np.zeros(M+N-1, dtype="int")  # Initialize result vector of convolution
for n in range(0, M+N-1):  # end vector will be M+N-1 # n = [0, 1, 2, ..., M+N-2]
    for m in range(0, M):  # m = [0, 1, 2, ..., M-1]
        if n-m < N and n-m >= 0:  # The signal s is considered zero outside its support
            glc[n] += h[m] * s[(n - m)] 
print(f'own linear conv:                    = {glc}')

# Numpy's implementation of the linear convolution
glc = np.convolve(s, h)  
print(f'numpy linear conv.:                 = {glc}')

# Own implementation of the circular convolution
gcc = np.zeros(N, dtype="int") # Initialize result vector of convolution
for n in range(0, N):  # n = [0, 1, 2, ..., 2*M-2]
    for m in range(0, M):  # m = [0, 1, 2, ..., M-1]
        if -N <= n-m and n-m < N:
            gcc[n] += h[m] * s[(n - m)]  # Python creates circular replication by its own
print(f'own circular conv.:                 = {gcc}')

# ifft and fft to implement the circular convolution
lh = np.shape(h)[0]
ls = np.shape(s)[0]
# perform a minimal zero padding so that the signal vectors h and s have equal lenghts
lpad = np.max((lh, ls))
hp = np.pad(h, (0, lpad-lh))
sp = np.pad(s, (0, lpad-ls))
gcc = np.round(np.real(ifft(fft(sp)*fft(hp)))).astype(np.int8)
print(f'circular conv.: ifft(fft(sp)*fft(hp)) = {gcc}')

# Zero padding to make circular convolution identical to linear convolution
lh = np.shape(h)[0]
ls = np.shape(s)[0]
lsoll = lh+ls-1
hp = np.pad(h, (0, lsoll-lh), "constant")
sp = np.pad(s, (0, lsoll-ls), "constant")
gccp = np.round(np.real(ifft(fft(sp)*fft(hp)))).astype(np.int8)
print(f'circular conv. zero padded:         = {gccp}')