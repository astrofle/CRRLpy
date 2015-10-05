#!/usr/bin/env python

import numpy as np

def draw_beam(header, ax):
    """
    Draws an elliptical beam
    in a pywcsgrid2 axes object.
    """
    
    bmaj = header.get("BMAJ")
    bmin = header.get("BMIN")
    pa = header.get("BPA")
    pixx = header.get("CDELT1")
    pixy = header.get("CDELT2")
    ax.add_beam_size(bmaj/np.abs(pixx), bmin/np.abs(pixy), pa, loc=3)