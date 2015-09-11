#!/usr/bin/env python

import numpy as np
from numpy.polynomial.polynomial import polyval

def make_ripples(n, b, nout, rms):
    
    if n < nout:
        print "Requested number of channels is larger than box size."
        print "Will not generate ripples."
        return np.zeros(len(nout))
    
    box = np.zeros(n)
    box[b/2:-b/2] = 1
    base = np.fft.fft(box)
    
    return rms*base.real[n/2-nout/2:n/2+nout/2]

def make_noise(mu, sigma, n):
    
    return np.random.normal(mu, sigma, n)


def make_offset(rms, x, order=0):
    """
    Generates an offset.
    Order specifies the 
    polynomial order to use.
    """
    
    c = []
    for i in range(order):
        if i == 0:
            c.append(np.random.normal(0, 2*rms))
        else:
            c.append(np.random.uniform(-1, 1))
    
    return 100*rms*polyval((x - x.mean())/x.mean(), c, tensor=False)