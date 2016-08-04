#!/usr/bin/env python

import numpy as np

class Spectrum(object):
    """
    """
    
    def __init__(self, x, y, w=None, spw=None, stack=None):
        
        # User provided values
        self.x = np.ma.masked_invalid(x)
        self.y = np.ma.masked_invalid(y)
        if not w:
            self.z = np.ma.masked_invalid(np.ones(len(x)))
        else:
            self.z = np.ma.masked_invalid(w)
        self.z.fill_value = 1.
        self.spw = spw
        self.stack = stack
        
        # Determine the global mask from x, y and z
        self.mask = self.x.mask | self.y.mask | self.z.mask
        # Apply the global mask to the data
        self.x.mask = self.mask
        self.y.mask = self.mask
        self.z.mask = self.mask
    
    def cut_edges(self, redge, ledge=None):
        """
        Mask the edges of the Spectrum frequency and amplitude.
        """
        
        if ledge == None:
            ledge = redge
            
        if isinstance(redge, int) and isinstance(ledge, int):
            self.x.mask[:ledge] = True
            self.x.mask[-redge:] = True
            self.y.mask[:ledge] = True
            self.y.mask[-redge:] = True
        else:
            raise ValueError('redge and ledge should be integers')
    
    def mask_ranges(self, ranges):
        """
        Masks the Spectrum inside the given ranges.
        
        :param ranges: Indexes defining the ranges to be masked.
        :type ranges: list of tuples
        """
        
        for rng in ranges:
            self.x.mask[rng[0]:rng[1]] = True
            self.y.mask[rng[0]:rng[1]] = True
            self.z.mask[rng[0]:rng[1]] = True
    
    def split_lines(self):
        """
        Splits the spectrum to separate lines.
        """
        
    def bandpass_corr(self, order):
        """
        """
        
        # Add offset to avoid zeros
        off = 1000.
        oy = self.y.compressed() + off
        
        # Use a polynomial to remove the baseline
        bp = np.polynomial.polynomial.polyfit(self.x.compressed(), oy, order, w=self.z)
        # Interpolate and extrapolate to the original x axis
        b = np.polynomial.polynomial.polyval(self.x.data, bp)
        
        # Flag NaN values in the baseline
        mb = np.ma.masked_invalid(b)
        mb.fill_value = 1.
        self.mb = np.ma.masked_equal(mb, 0)
        
        self.ycorr = np.ma.masked_invalid((oy/mb.filled() - 1.)*off)
                
if __name__ == '__main__':
    pass