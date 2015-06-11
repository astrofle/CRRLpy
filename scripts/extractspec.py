#!/usr/bin/env python

import sys
import re

from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii
from astropy import wcs
from astropy.coordinates import SkyCoord

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
    
    if shape == 'point':
        # Convert sky coordinates to pixels if required
        if 'sky' in coord.lower():
            
            coo_sky = SkyCoord(params[0], params[1], frame='fk5')
            
            params[0:2] = w.all_world2pix([[coo_sky.ra.value, 
                                            coo_sky.dec.value]], 0)[0]

        params = [int(round(float(x))) for x in params]
        rgn = {'shape':'point',
               'params':{'cx':params[0], 'cy':params[1]}}
    
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
                         'trcx':params[2], 'trcy':params[3]}}
               
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
               'params':{'cx':params[0], 'cy':params[1], 'r':params[2]}}
        
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
    
    elif rgn['shape'] == 'circle':
        patch = mpatches.Circle((rgn['params']['cx'], rgn['params']['cy']), 
                                rgn['params']['r'], alpha=0.5, transform=ax.transData)
        #plt.figure().artists.append(patch)
        ax.add_patch(patch)

def split_str(str):
    """
    Splits text from digits in a string.
    """
    
    match = re.match(r"([0-9]+)([a-z]+)", str, re.I)
    
    if match:
        items = match.groups()
        
    return items[0], items[1]
    
    
def main(out, cube, region):
    """
    """
    
    hdulist = fits.open(cube)
    head = hdulist[0].header
    data = hdulist[0].data
    
    #blank = head['BLANK']
    
    data = np.ma.masked_invalid(data)
    
    # Build a WCS object to handle sky coordinates
    w = set_wcs(head)
    
    rgn = parse_region(region, w)
    
    # Get the frequency axis
    freq = get_axis(head, 3)

    spec = extract_spec(data, rgn)
    
    #spec.set_fill_value(blank)
    #print "nspec {0}".format(len(spec))
    tbtable = Table([freq, spec.filled()], 
                    names=['{0} {1}'.format(head['CTYPE3'], head['CUNIT3']),
                           'Tb {0}'.format(head['BUNIT'])])

    ascii.write(tbtable, out, format='commented_header')
    
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data.sum(axis=0).sum(axis=0), origin='lower', interpolation='none')
    show_rgn(ax, rgn)

    plt.savefig('{0}_extract_region_{1}.png'.format(cube, region))#, 
                #bbox_inches='tight', pad_inches=0.3)

if __name__ == '__main__':
    
    cube = sys.argv[1]
    region = sys.argv[2]
    out = sys.argv[3]
    
    main(out, cube, region)
    