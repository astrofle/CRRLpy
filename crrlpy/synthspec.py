#!/usr/bin/env python

import numpy as np
from crrlpy import crrls

def make_noise(mu, sigma, n):
    
    return np.random.normal(mu, sigma, n)

def make_baseline(n, b):
    
    box = np.zeros(n)
    box[b/2:-b/2] = 1
    base = np.fft.fft(box)
    
    #return np.abs(base)
    return base.real

def make_lw(n, freq, dn, Te, ne, Tr, W, dD):
    """
    Freq in Hz
    Doppler width dD in m s^-1
    Returns the combined line width in Hz
    """
    
    dP = crrls.pressure_broad(n, Te, ne) # kHz
    dR = crrls.radiation_broad(n, W, Tr) # kHz
    
    # Convert Doppler line width to frequency units
    dfD = crrls.dv2df(freq, dD) # Hz
    dfD = dfD/1e3               # kHz

    # Make the combined linewidths
    dL = dP + dR
    df = crrls.line_width(dfD, dL)
    
    return df*1e3

def make_dL(n, Te, ne, Tr, W):
    
    dP = crrls.pressure_broad(n, Te, ne) # kHz
    dR = crrls.radiation_broad(n, W, Tr) # kHz
    
    dL = dP + dR
    
    return dL