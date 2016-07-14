#!/usr/bin/env python

import numpy as np

class Spectrum(object):
    """
    """
    
    def __init__(self, x, y, spw=None, stack=None, is_vel=False, n=None):
        
        # User provided values
        self.freq = np.ma.masked_invalid(x)
        self.amp = np.ma.masked_invalid(y)
        
        if is vel:
            self.has_freq = False
            self.freq = np.ma.ones(len(x))
        
        self.spw = spw
        self.stack = stack
        self.has_vel = is_vel
        
        # Determine the global mask from freq and amp
        self.mask = self.freq.mask | self.amp.mask
        # Apply the global mask to the data
        self.freq.mask = self.mask
        self.amp.mask = self.mask
    
    def cut_edges(self, redge, ledge=0):
        """
        Mask the edges of a Spectrum frequency and amplitude
        """
    
    def get_umask_amp(self):
        """
        Returns the unmasked elements of amp.
        """
        
        umask = self.amp.nonzero()
        return self.amp[umask]
    
    def get_umask_freq(self):
        """
        Returns the unmasked elements of freq.
        """
        
        umask = self.freq.nonzero()
        return self.freq[umask]
        
def bandpass_corr(x, y, order, median=False):
    """
    """
    
    # Turn NaNs to zeros
    my = np.ma.masked_invalid(y)
    mx = np.ma.masked_where(np.ma.getmask(my), x)
    mmx = np.ma.masked_invalid(mx)
    mmy = np.ma.masked_where(np.ma.getmask(mmx), my)
    np.ma.set_fill_value(mmy, 0)
    np.ma.set_fill_value(mmx, 0)
    gx = mmx.compressed()
    gy = mmy.compressed()
    
    # Use a polynomial to remove the baseline
    #print gx
    #print gy
    bp = np.polynomial.polynomial.polyfit(gx, gy, order)
    # Interpolate and extrapolate to the original x axis
    b = np.polynomial.polynomial.polyval(x, bp)
    
    # Flag NaN values in the baseline
    mb = np.ma.masked_invalid(b)
    mb.fill_value = 0.0
    
    if median:
        # Only keep the baseline shape
        gb = mb - np.median(mb.compressed())
        #print np.median(mb.compressed())
    else:
        gb = mb
    
    ycorr = y - gb
            
    return ycorr, gb

if __name__ == '__main__':
    pass