#!/usr/bin/env python

"""
Extracts a spectrum from a region in a spectral cube.
The pixels will be averaged spatially inside the given region.
The region must be specified as shape,coords,parameters
Region can be, point, box or circle.
Coords can be pix or sky.
Parameters, for point the coordinates of the point, e.g. point,pix,512,256.
For box the bottom left corner and top right corner coordinates, e.g. box,pix,256,256,512,512.
For circle the center of the circle and the radius, e.g. circle,pix,512,256,10.
If the coordinates are given in sky values, then the units must included, e.g. 
point,pix,12h,2d will extract the spectrum for the pixel located at RA 12 hours and DEC 2 degrees.
For circle the radius units in sky coordinates can be either d for degrees, m for arcminutes or 
s for arcseconds. The conversion will use the scale in the RA direction of the cube, i.e. will 
use CDELT1 to convert from angular units to pixels.
"""

import sys
import re
import argparse

from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii
from astropy import wcs
from astropy.coordinates import SkyCoord
from crrlpy.crrls import is_number

import astropy.units as u
import matplotlib.patches as mpatches
import numpy as np
import pylab as plt

def sector_mask(shape, centre, radius, angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx, y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def parse_region(region, w):
    """
    Parses a region description to a dict
    """
    
    shape = region.split(',')[0]
    #print shape
    coord = region.split(',')[1]
    params = region.split(',')[2:]
    
    if is_number(params[-1]):
        label = None 
    else:
        params = params[:-1]
        label = region.split(',')[-1]
    
    if shape == 'point':
        # Convert sky coordinates to pixels if required
        if 'sky' in coord.lower():
            
            coo_sky = SkyCoord(params[0], params[1], frame='fk5')
            
            params[0:2] = w.all_world2pix([[coo_sky.ra.value, 
                                            coo_sky.dec.value]], 0)[0]

        params = [int(round(float(x))) for x in params]
        rgn = {'shape':'point',
               'params':{'cx':params[0], 'cy':params[1]},
               'label':label}
    
    elif shape == 'box':
        # Convert sky coordinates to pixels if required
        if 'sky' in coord.lower():
            
            blc_sky = SkyCoord(params[0], params[1], frame='fk5')
            trc_sky = SkyCoord(params[2], params[3], frame='fk5')
            
            params[0:2] = w.all_world2pix([[blc_sky.ra.value, 
                                            blc_sky.dec.value]], 0)[0]
            params[2:] = w.all_world2pix([[trc_sky.ra.value, 
                                           trc_sky.dec.value]], 0)[0]
        
        params = [int(round(float(x))) for x in params]
        rgn = {'shape':'box',
               'params':{'blcx':params[0], 'blcy':params[1], 
                         'trcx':params[2], 'trcy':params[3]},
               'label':label}
               
    elif shape == 'circle':
        # Convert sky coordinates to pixels if required
        if 'sky' in coord.lower():
            
            coo_sky = SkyCoord(params[0], params[1], frame='fk5')
            
            params[0:2] = w.all_world2pix([[coo_sky.ra.value, 
                                            coo_sky.dec.value]], 0)[0]
            
            lscale = abs(w.to_fits()[0].header['CDELT1'])*u.deg
            val, uni = split_str(params[2])
            
            # Add units to the radius
            if uni == 'd':
                r = val*u.deg
            if uni == 'm':
                r = val*u.arcmin
            if uni == 's':
                r = val*u.arcsec
                
            params[2] = r/lscale
        
        params = [int(round(float(x))) for x in params]
        rgn = {'shape':'circle',
               'params':{'cx':params[0], 'cy':params[1], 'r':params[2]},
               'label':label}
        
    else:
        print 'region description not supported.'
        print 'Will exit now.'
        sys.exit()
        
    return rgn

def get_axis(header, axis):
    """
    Constructs a cube axis
    @param header - fits cube header
    @type header - pyfits header
    @param axis - axis to reconstruct
    @type axis - int
    @return - cube axis
    @rtype - numpy array
    """
    
    axis = str(axis)
    dx = header.get("CDELT" + axis)
    dx = int(dx)
    p0 = header.get("CRPIX" + axis)
    x0 = header.get("CRVAL" + axis)
    n = header.get("NAXIS" + axis)
    print "Number channels in extracted spectrum: {0}".format(n)
    
    return np.arange(x0 - p0*dx, x0 - p0*dx + n*dx, dx)

def extract_spec(data, region):
    """
    Sums the pixels inside a region preserving the spectral axis.
    """
    
    if region['shape'] == 'point':
        spec = data[:,:,region['params']['cy'],region['params']['cx']]
        spec = spec.sum(axis=0)
    
    elif region['shape'] == 'box':
        spec = data[:,:,region['params']['blcy']:region['params']['trcy'],
                    region['params']['blcx']:region['params']['trcx']]
        area = (region['params']['trcy'] - region['params']['blcy']) * \
               (region['params']['trcx'] - region['params']['blcx'])
        spec = spec.sum(axis=3).sum(axis=2).sum(axis=0)/area
        
    elif region['shape'] == 'circle':
        mask = sector_mask(data[0,0].shape,
                           (region['params']['cy'], region['params']['cx']),
                           region['params']['r'],
                           (0, 360))
        
        # Mask the data ouside the circle
        # This method is too slow
        #mdata = np.ma.empty((data.sum(axis=0).shape))
        #for c in xrange(len(mdata)):
            #mdata[c] = np.ma.masked_where(~mask, data.sum(axis=0)[c])
        #spec = mdata.sum(axis=2).sum(axis=1)/(mdata.count()/len(mdata))
       
        mdata = data.sum(axis=0)[:,mask]
        spec = mdata.sum(axis=1)/len(np.where(mask.flatten()==1)[0])
        
    return spec

def set_wcs(head):
    """
    Build a WCS object given the 
    spatial header parameters.
    """
    
    # Create a new WCS object.  The number of axes must be set
    # from the start.
    w = wcs.WCS(naxis=2)
    
    w.wcs.crpix = [head['CRPIX1'], head['CRPIX2']]
    w.wcs.cdelt = [head['CDELT1'], head['CDELT2']]
    w.wcs.crval = [head['CRVAL1'], head['CRVAL2']]
    w.wcs.ctype = [head['CTYPE1'], head['CTYPE2']]
    
    return w

def show_rgn(ax, rgn):
    """
    Plots the extraction region.
    """
    
    if rgn['shape'] == 'box':
        ax.plot([rgn['params']['blcx']]*2, 
                 [rgn['params']['blcy'],rgn['params']['trcy']], 'r-')
        ax.plot([rgn['params']['blcx'],rgn['params']['trcx']], 
                 [rgn['params']['blcy']]*2, 'r-')
        ax.plot([rgn['params']['blcx'],rgn['params']['trcx']], 
                 [rgn['params']['trcy']]*2, 'r-')
        ax.plot([rgn['params']['trcx']]*2, 
                 [rgn['params']['blcy'],rgn['params']['trcy']], 'r-')
        # Define label location
        xlabel = (rgn['params']['trcx'] + rgn['params']['blcx'])/2.
        ylabel = rgn['params']['trcy']
    
    elif rgn['shape'] == 'circle':
        patch = mpatches.Circle((rgn['params']['cx'], rgn['params']['cy']), 
                                rgn['params']['r'], alpha=1, 
                                transform=ax.transData, label=rgn['label'],
                                color='red', ec='r', fc=None, lw=2, fill=False)
        ax.add_patch(patch)
        # Define label location
        xlabel = rgn['params']['cx'] #- 0.5*rgn['params']['r']
        ylabel = rgn['params']['cy'] + rgn['params']['r']
        
    ax.annotate(rgn['label'], xy=(xlabel, ylabel), 
                xytext=(xlabel, ylabel+0.2), color='r')

def split_str(str):
    """
    Splits text from digits in a string.
    """
    
    match = re.match(r"([0-9]+)([a-z]+)", str, re.I)
    
    if match:
        items = match.groups()
        
    return items[0], items[1]
    
    
def main(out, cube, regions):
    """
    """
    
    hdulist = fits.open(cube)
    head = hdulist[0].header
    data = hdulist[0].data
    
    #blank = head['BLANK']
    
    data = np.ma.masked_invalid(data)
    
    # Build a WCS object to handle sky coordinates
    #w = set_wcs(head)
    w=None
    
    rgns = np.empty(len(regions.split('/')), dtype=object)
    
    for i,region in enumerate(regions.split('/')):
        rgns[i] = parse_region(region, w)
    
    # Get the frequency axis
    #freq = get_axis(head, 3)

    #spec = extract_spec(data, rgn)
    
    #spec.set_fill_value(blank)
    #print "nspec {0}".format(len(spec))
    #try:
        #freqvvel = head['CUNIT3']
    #except KeyError:
        #freqvvel = 'KMS'
    #tbtable = Table([freq, spec.filled()], 
                    #names=['{0} {1}'.format(head['CTYPE3'], freqvvel),
                           #'Tb {0}'.format(head['BUNIT'])])

    #ascii.write(tbtable, out, format='commented_header')
    
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    try:
        ax.imshow(data.sum(axis=0).sum(axis=0), interpolation='none', origin='lower')
    except TypeError:
        ax.imshow(data.sum(axis=0), interpolation='none', origin='lower')
    
    for i,region in enumerate(regions.split('/')):
        show_rgn(ax, rgns[i])
        
    ax.set_xlim(-0.5, head['NAXIS1']-0.5)
    ax.set_ylim(-0.5, head['NAXIS2']-0.5)
    
    plt.savefig('{0}'.format(out), 
                bbox_inches='tight', pad_inches=0.3)
    plt.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('cube', type=str,
                        help="Cube to extract the spectrum from. (string)")
    parser.add_argument('region', type=str,
                        help="Spatial region where pixels are averaged. " \
                             "e.g., point,sky,18h46m22s,-2d56m12s")
    parser.add_argument('out', type=str,
                        help="Output figure name.")
    args = parser.parse_args()
    
    main(args.out, args.cube, args.region)
    