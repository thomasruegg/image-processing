#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Paper Exercise Lab 10

Image Reconstruction Filters

@author: Patrik Müller
@date:   21.11.2019
"""

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def get_mean_filter_coeff(sz):
    return np.ones((sz,sz))/(sz*sz)

def arit_mean_filter(sz, f):
    g = get_mean_filter_coeff(sz)
    return convolve2d(f, g, mode='same')

def harmonic_mean_filter(sz, f):
    h = np.zeros_like(f)
    f = np.pad(f, sz//2, 'constant')
    mn = sz*sz
    for x in range(h.shape[0]):
        for y in range(h.shape[1]):
            h[x,y] = mn / np.sum(1 / (f[x:x+sz, y:y+sz] + 1e-12)) # adding 1e-12 for numerical stability
    return h

def contraharmonic_mean_filter(sz, f, Q=-1):
    h = np.zeros_like(f)
    f = np.pad(f, sz//2, 'constant')
    for x in range(h.shape[0]):
        for y in range(h.shape[1]):
            g = f[x:x+sz, y:y+sz] + 1e-12 # adding 1e-12 for numerical stability
            h[x,y] = np.sum(g**(Q + 1)) / np.sum(g**Q)
    return h

def max_filter(sz, f):
    h = np.zeros_like(f)
    f = np.pad(f, sz//2, 'constant')
    for x in range(h.shape[0]):
        for y in range(h.shape[1]):
            h[x,y] = np.max(f[x:x+sz, y:y+sz])
    return h

def midpoint_filter(sz, f): 
    h = np.zeros_like(f)
    f = np.pad(f, sz//2, 'constant')
    for x in range(h.shape[0]):
        for y in range(h.shape[1]):
            g = f[x:x+sz, y:y+sz]
            h[x,y] = 0.5*(g.min() + g.max())
    return h

def display_image(f):
    plt.imshow(f, cmap='gray', vmin=0, vmax=255)

def main():
    bwidth = 7
    bspacing = 17
    border = 20
    h = 210
    w = 9*bwidth+8*bspacing
    
    filter_sizes = [3, 5, 7, 9]
    filters = [arit_mean_filter, harmonic_mean_filter, contraharmonic_mean_filter, 
               max_filter, midpoint_filter]
    filter_names = ['Arithmetic Mean', 'Harmonic Mean', 'Contraharmonic Mean', 
                    'Max', 'Midpoint']
    
    f = np.zeros((h+2*border,w+2*border))
    
    # draw image
    for i in range(9):
        xstart = i*(bwidth+bspacing)+border
        f[border:-border, xstart:xstart+bwidth] = 255
      
    n_sz = len(filter_sizes)
    y = int(np.ceil(n_sz / 2))
    x = int(np.ceil(n_sz / y))
    
    # Calculate filtered images and display
    for i, filt in enumerate(filters):
        print('Calculating {} filter'.format(filter_names[i]))
        fig = plt.figure(i+1)
        fig.suptitle(filter_names[i])
        for j, sz in enumerate(filter_sizes):
            plt.subplot(y,x,j+1)
            plt.title('filter size = {}x{}'.format(sz,sz))
            display_image(filt(sz,f))
        
    plt.show()

if __name__=='__main__':
    main()