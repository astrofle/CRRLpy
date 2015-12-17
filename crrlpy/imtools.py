#!/usr/bin/env python

import numpy as np

from matplotlib import _cntr as cntr

def draw_beam(header, ax):
    """
    Draws an elliptical beam in a pywcsgrid2 axes object.
    """
    
    bmaj = header.get("BMAJ")
    bmin = header.get("BMIN")
    pa = header.get("BPA")
    pixx = header.get("CDELT1")
    pixy = header.get("CDELT2")
    ax.add_beam_size(bmaj/np.abs(pixx), bmin/np.abs(pixy), pa, loc=3)
    
def get_contours(x, y, z, levs, segment=0, verbose=False):
    """
    Creates an array with the contour vertices.
    """
    
    c = cntr.Cntr(x, y, z)
    
    segments = []
    for i,l in enumerate(levs):
        res = c.trace(l)
        if res:
            nseg = len(res) // 2
            segments.append(res[:nseg][segment])
            if verbose: print res[:nseg][segment]
        else:
            pass
                
    return np.asarray(segments)

def remove_nans(contours, indx=0):
    """
    Removes NaN elements from a contour list produced with get_contours().
    """
    
    mask = np.isnan(contours[indx][:,0])
    contours[indx] = contours[indx][~mask]
    
    return contours