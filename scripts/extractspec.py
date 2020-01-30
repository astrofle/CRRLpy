#!/usr/bin/env python3

"""
Extracts a spectrum from a region in a spectral cube.
The pixels will be averaged spatially inside the given region.
The region must be specified as shape,coords,frame,parameters
Region can be, point, box, circle or ellipse. Also, CASA region files are supported.
Coords can be pix or sky.
Frames supported include: fk4, fk5, icrs, and gal.
Parameters, for point the coordinates of the point, e.g. point,pix,,512,256.
For box the bottom left corner and top right corner coordinates, e.g. box,pix,,256,256,512,512.
For circle the center of the circle and the radius, e.g. circle,pix,,512,256,10.
For ellipse, its center, major and minor axes and the angle in degrees, e.g., 
ellipse,sky,fk5,13h05m27.462s,-49d28m06.547s,0.345293s,0.493276s,0
If the coordinates are given in sky values, then the units must included, e.g. 
point,sky,12h,2d will extract the spectrum for the pixel located at RA 12 hours and DEC 2 degrees.
For circle the radius units in sky coordinates can be either d for degrees, m for arcminutes or 
s for arcseconds. The conversion will use the scale in the RA direction of the cube, i.e. will 
use CDELT1 to convert from angular units to pixels.
"""

import os
import matplotlib as mpl
havedisplay = "DISPLAY" in os.environ
if not havedisplay:
    mpl.use('Agg')

import sys
import re
import argparse
import logging

from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

import crrlpy.imtools as ci
import astropy.units as u
import matplotlib.patches as mpatches
import numpy as np
import pylab as plt

eq_frames = ['icrs', 'fk5', 'fk4']

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

def ellipse_mask(shape, x0, y0, bmaj, bmin, angle):
    """
    """
    
    x,y = np.mgrid[:shape[0],:shape[1]]
    
    cos_angle = np.cos(np.radians(180. - angle))
    sin_angle = np.sin(np.radians(180. - angle))
    
    # Shift the indexes.
    xc = x - x0
    yc = y - y0
    
    # Rotate the indexes.
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle - yc * cos_angle
    
    mask = (xct/bmaj)**2. + (yct/bmin)**2. <= 1.
    
    return mask

def add_radius_units(value, units):
    """
    """
    
    if units == 'd':
        r = value*u.deg
    if units == 'm':
        r = value*u.arcmin
    if units == 's':
        r = value*u.arcsec
        
    return r

def parse_region(region, wcs):
    """
    Parses a region description to a dict
    """
    
    logger = logging.getLogger(__name__)
    
    shape = region.split(',')[0]
    coord = region.split(',')[1]
    frame = region.split(',')[2]
    try:
        params = region.split(',')[3:]
    except IndexError:
        logger.error('No coordinates given.')
        logger.error('Will exit now.')
        sys.exit(1)
        
    if 'sky' in coord.lower() and frame == '':
        logger.error('No frame specified.')
        logger.error('Will exit now.')
        sys.exit(1)
    
    if shape == 'point':
        # Convert sky coordinates to pixels if required
        if 'sky' in coord.lower():
            
            coo_sky = SkyCoord(params[0], params[1], frame=frame)
            
            if frame in eq_frames:
                params[0:2] = wcs.all_world2pix([[coo_sky.ra.value, 
                                                  coo_sky.dec.value]], 0)[0]
            elif 'gal' in frame:
                params[0:2] = wcs.all_world2pix([[coo_sky.l.value, 
                                                  coo_sky.b.value]], 0)[0]

        params = [int(round(float(x))) for x in params]
        rgn = {'shape':'point',
               'params':{'cx':params[0], 'cy':params[1]}}
    
    elif shape == 'box':
        # Convert sky coordinates to pixels if required
        if 'sky' in coord.lower():
            
            blc_sky = SkyCoord(params[0], params[1], frame=frame)
            trc_sky = SkyCoord(params[2], params[3], frame=frame)
            
            if frame in eq_frames:
                params[0:2] = wcs.all_world2pix([[blc_sky.ra.value, 
                                                blc_sky.dec.value]], 0)[0]
                params[2:] = wcs.all_world2pix([[trc_sky.ra.value, 
                                               trc_sky.dec.value]], 0)[0]
            elif 'gal' in frame:
                params[0:2] = wcs.all_world2pix([[blc_sky.l.value, 
                                                blc_sky.b.value]], 0)[0]
                params[2:] = wcs.all_world2pix([[trc_sky.l.value, 
                                               trc_sky.b.value]], 0)[0]
                
        params = [int(round(float(x))) for x in params]
        rgn = {'shape':'box',
               'params':{'blcx':params[0], 'blcy':params[1], 
                         'trcx':params[2], 'trcy':params[3]}}
               
    elif shape == 'circle':
        # Convert sky coordinates to pixels if required
        if 'sky' in coord.lower():
            
            coo_sky = SkyCoord(params[0], params[1], frame=frame)
            
            if frame in eq_frames:
                params[0:2] = wcs.all_world2pix([[coo_sky.ra.value, 
                                                  coo_sky.dec.value]], 0)[0]
            elif 'gal' in frame:
                params[0:2] = wcs.all_world2pix([[coo_sky.l.value, 
                                                  coo_sky.b.value]], 0)[0]
            
            lscale = abs(wcs.pixel_scale_matrix[0,0])*u.deg
            val, uni = split_str(params[2])
            
            # Add units to the radius
            r = add_radius_units(val, uni)
            logger.debug('lscale: {0}'.format(lscale))
            logger.debug('radius: {0}'.format(r))
            params[2] = (r/lscale).cgs.value
        
        params = [float(x) for x in params]
        rgn = {'shape':'circle',
               'params':{'cx':params[0], 'cy':params[1], 'r':params[2]}}
    
    elif shape == 'ellipse':
        # Convert sky coordinates to pixels if required
        if 'sky' in coord.lower():
            
            coo_sky = SkyCoord(params[0], params[1], frame=frame)
            
            if frame in eq_frames:
                params[0:2] = wcs.all_world2pix([[coo_sky.ra.value, 
                                                  coo_sky.dec.value]], 0)[0]
            elif 'gal' in frame:
                params[0:2] = wcs.all_world2pix([[coo_sky.l.value, 
                                                  coo_sky.b.value]], 0)[0]
            
            lscale = abs(wcs.pixel_scale_matrix[0,0])*u.deg
            logger.debug('lscale: {0}'.format(lscale))
            
            # Major axis.
            val, uni = split_str(params[2])
            # Add units to the major axis.
            r = add_radius_units(val, uni)
            logger.debug('major axis: {0}'.format(r))
            params[2] = (r/lscale).cgs.value
            
            # Minor axis.
            val, uni = split_str(params[3])
            # Add units to the minor axis.
            r = add_radius_units(val, uni)
            logger.debug('minor axis: {0}'.format(r))
            params[3] = (r/lscale).cgs.value
        
        params = [float(x) for x in params]
        rgn = {'shape':'ellipse',
               'params':{'cx':params[0], 'cy':params[1], 
                         'bmaj':params[2], 'bmin':params[3], 
                         'theta':params[4]}}
    
    elif shape == 'crtf':
        # CASA region files are always in sky coordinates
        polys = ci.read_casa_polys(params[0], wcs=wcs)
        rgn = {'shape':'polygon',
               'params':{'Polygons':polys}}
        
    elif shape == 'all':
        rgn = {'shape':'all',
               'params':'all'}
        
    else:
        print('region description not supported.')
        print('Will exit now.')
        logger.error('Region description not supported.')
        logger.error('Will exit now.')
        sys.exit(1)
        
    return rgn

def plotspec(cube, faxis, taxis, ftype, funit, tunit, out):
    """
    """
    
    logger = logging.getLogger(__name__)
    logger.info("Plotting extracted spectrum to {0}".format(out))
    
    fig = plt.figure(frameon=False)
    fig.suptitle(cube)
    ax = fig.add_subplot(1, 1, 1)
     
    ax.plot(faxis, taxis, 'k-', drawstyle='steps')
    
    ax.set_xlabel('{0} axis ({1})'.format(ftype, funit))
    ax.set_ylabel('Temperature axis ({0})'.format(tunit))
    
    plt.savefig('{0}'.format(out), 
                bbox_inches='tight', pad_inches=0.3)
    plt.close()

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
    
    logger = logging.getLogger(__name__)
    
    wcs = WCS(header)
    
    # swapaxes uses python convention for axes index.
    wcs = wcs.swapaxes(axis-1,0)
    wcs = wcs.sub(1)
    n_axis = wcs.array_shape[-axis]
    axis_vals = wcs.pixel_to_world_values(np.arange(0,n_axis))[0]
    
    return axis_vals

def extract_spec(data, region, naxis, mode):
    """
    Sums the pixels inside a region preserving the spectral axis.
    """
    
    logger = logging.getLogger(__name__)
    
    logger.debug('Data shape: {0}'.format(data.shape))
    
    if region['shape'] == 'point':
        if naxis > 3:
            spec = data[:,:,region['params']['cy'],region['params']['cx']]
            if mode == 'sum':
                spec = spec.sum(axis=0)
            elif mode == 'avg':
                spec = spec.mean(axis=0)
            elif 'flux' in mode.lower():
                spec = spec.sum(axis=0)/region['barea']
            else:
                logger.error('Mode not supported.')
                logger.error('Will exit now.')
                sys.exit(1)
        elif naxis == 3:
            spec = data[:,region['params']['cy'],region['params']['cx']]
        else:
            spec = data[region['params']['cy'],region['params']['cx']]
            
    elif region['shape'] == 'box':
        area = (region['params']['trcy'] - region['params']['blcy']) * \
                (region['params']['trcx'] - region['params']['blcx'])
        
        if naxis > 3:
            spec = data[0,:,region['params']['blcy']:region['params']['trcy'],
                        region['params']['blcx']:region['params']['trcx']]
            
            if mode == 'sum':
                spec = spec.sum(axis=2).sum(axis=1)
            elif mode == 'avg':
                spec = spec.mean(axis=2).mean(axis=1)
            elif 'flux' in mode.lower():
                spec = spec.sum(axis=2).sum(axis=1)/region['barea']
            else:
                logger.error('Mode not supported.')
                logger.error('Will exit now.')
                sys.exit(1)
                
        elif naxis == 3:
            spec = data[:,region['params']['blcy']:region['params']['trcy'],
                        region['params']['blcx']:region['params']['trcx']]
            if mode == 'sum':
                spec = spec.sum(axis=2).sum(axis=1)#/area
            elif mode == 'avg':
                spec = spec.mean(axis=2).mean(axis=1)#/area
            elif 'flux' in mode.lower():
                summ = spec.sum(axis=2).sum(axis=1)
                logger.info('Sum of pixels: {0}'.format(summ))
                spec = summ/region['barea']
            else:
                logger.error('Mode not supported.')
                logger.error('Will exit now.')
                sys.exit(1)
                
        else:
            spec = data[region['params']['blcy']:region['params']['trcy'],
                        region['params']['blcx']:region['params']['trcx']]
            if mode == 'sum':
                spec = spec.sum()
            elif mode == 'avg':
                spec = spec.mean()
            elif 'flux' in mode.lower():
                spec = spec.sum()/region['barea']
            else:
                logger.error('Mode not supported.')
                logger.error('Will exit now.')
                sys.exit(1)
                
    elif region['shape'] == 'circle':
        logger.info("Circular region has a center " \
                    "at pixel ({0},{1}) with radius " \
                    "{2}".format(region['params']['cx'], 
                                    region['params']['cy'], 
                                    region['params']['r']))
                    
        if naxis > 3:
            logger.debug("The image has more than 3 axes.")
            mask = sector_mask(data[0,0].shape,
                               (region['params']['cy'], region['params']['cx']),
                               region['params']['r'],
                               (0, 360))
            mdata = data[0][:,mask]
            logger.debug("Masked data shape: {0}".format(mdata.shape))
            if 'sum' in mode.lower():
                spec = mdata.sum(axis=1)
            elif 'avg' in mode.lower():
                spec = mdata.mean(axis=1)
            elif 'flux' in mode.lower():
                spec = mdata.sum(axis=1)/region['barea']
                
        elif naxis == 3:
            
            mask = sector_mask(data[0].shape,
                               (region['params']['cy'], region['params']['cx']),
                               region['params']['r'],
                               (0, 360))
            mdata = data[:,mask]
            logger.debug("Masked data shape: {0}".format(mdata.shape))
            if 'sum' in mode.lower():
                spec = mdata.sum(axis=1)#/len(np.where(mask.flatten() == 1)[0])
            elif 'avg' in mode.lower():
                spec = mdata.mean(axis=1)
            elif 'flux' in mode.lower():
                spec = mdata.sum(axis=1)/region['barea']
            else:
                logger.error('Mode not supported.')
                logger.error('Will exit now.')
                sys.exit(1)
        
        else:
   
            mask = sector_mask(data.shape,
                               (region['params']['cy'], region['params']['cx']),
                               region['params']['r'],
                               (0, 360))
            mdata = np.ma.masked_invalid(data[mask])
            logger.debug("Masked data shape: {0}".format(mdata.shape))
            logger.debug("Masked data sum: {0}".format(mdata))
            if 'sum' in mode.lower():
                spec = mdata.sum()#/len(np.where(mask.flatten() == 1)[0])
            elif 'avg' in mode.lower():
                spec = mdata.mean()
            elif 'flux' in mode.lower():
                spec = mdata.sum()/region['barea']
            else:
                logger.error('Mode not supported.')
                logger.error('Will exit now.')
                sys.exit(1)
    
    elif region['shape'] == 'ellipse':
        logger.info("Elliptical region has a center " \
                    "at pixel ({0},{1}) with major and minor axes " \
                    "{2} and {3} at an angle {4}".format(region['params']['cx'], 
                                                         region['params']['cy'], 
                                                         region['params']['bmaj'],
                                                         region['params']['bmin'],
                                                         region['params']['theta']))
    
        logger.debug("Mask shape: {}".format(data.shape[-2:]))
        mask = ellipse_mask(data.shape[-2:],
                            region['params']['cy'], region['params']['cx'],
                            region['params']['bmaj']/2., region['params']['bmin']/2.,
                            region['params']['theta'])
        logger.debug('Elements in mask: {}'.format(mask.sum()))
        
        if naxis > 3:
            mdata = data[0][:,mask]
            axis = 1
        elif naxis == 3:
            mdata = data[:,mask]
            axis = 1
        else:
            mdata = data[mask]
            axis = 0
        logger.debug("Masked data shape: {0}".format(mdata.shape))
        
        if 'sum' in mode.lower():
            spec = mdata.sum(axis=axis)
        elif 'avg' in mode.lower():
            spec = mdata.mean(axis=axis)
        elif 'flux' in mode.lower():
            spec = mdata.sum(axis=axis)/region['barea']
        else:
            logger.error('Mode not supported.')
            logger.error('Will exit now.')
            sys.exit(1)
            
    elif 'poly' in region['shape']:
        npolys = len(region['params']['Polygons'])
        
        if naxis > 3:
            shape = data[0][0].shape
            npix3 = data[0].shape[0]
        elif naxis == 3:
            shape = data[0].shape
            npix3 = data.shape[0]
        else:
            shape = data.shape
            npix3 = 0
            
        mask = np.zeros(shape)
        
        for poly in region['params']['Polygons']:
            # Add all the polygons together
            logger.info("Adding polygons to the mask.")
            mask += poly.make_mask(shape)
            
        logger.info("Normalizing the mask to unity.")
        mask = np.ceil(mask/npolys)
        
        if naxis > 3:
            mdata = data[0]*np.tile(mask, (npix3,1,1))
        else:
            mdata = data*np.tile(mask, (npix3,1,1))
            
        if mode == 'sum':
            spec = mdata.sum(axis=1).sum(axis=1)
        elif 'avg' in mode.lower():
            spec = mdata.mean(axis=1).mean(axis=1)
        elif 'flux' in mode.lower():
            spec = mdata.sum(axis=1).sum(axis=1)/region['barea']
        else:
            logger.error('Mode not supported.')
            logger.error('Will exit now.')
            sys.exit(1)
            
    elif 'all' in region['shape']:
        
        if naxis > 3:
            data = data[0]
            spec = proc_data(data, mode, region)
        elif naxis == 3:
            data = data
            spec = proc_data(data, mode, region)
        else:
            spec = proc_data(data, mode, region)
        
    return spec

def proc_data(data, mode, region):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    if 'sum' in mode.lower():
            spec = data.sum(axis=1).sum(axis=1)
    elif 'avg' in mode.lower():
        spec = data.mean(axis=1).mean(axis=1)
    elif 'flux' in mode.lower():
        spec = data.sum(axis=1).sum(axis=1)/region['barea']
    else:
        logger.error('Mode not supported.')
        logger.error('Will exit now.')
        sys.exit(1)
    
    return spec

def set_wcs(head):
    """
    Build a WCS object given the 
    spatial header parameters.
    """
    
    logger = logging.getLogger(__name__)
    
    # Create a new WCS object. 
    wcs = WCS(head)
    
    if wcs.naxis > 3:
        wcs = wcs.dropaxis(2)

    logger.info('WCS contains {0} axes.'.format(wcs.naxis))
        
    return wcs

def show_rgn(ax, rgn, **kwargs):
    """
    Plots the extraction region.
    """
    
    alpha = 0.1
    #lw = 0.1
    
    if rgn['shape'] == 'box':
        ax.plot([rgn['params']['blcx']]*2, 
                 [rgn['params']['blcy'],rgn['params']['trcy']], 'r-', **kwargs)
        ax.plot([rgn['params']['blcx'],rgn['params']['trcx']], 
                 [rgn['params']['blcy']]*2, 'r-', **kwargs)
        ax.plot([rgn['params']['blcx'],rgn['params']['trcx']], 
                 [rgn['params']['trcy']]*2, 'r-', **kwargs)
        ax.plot([rgn['params']['trcx']]*2, 
                 [rgn['params']['blcy'],rgn['params']['trcy']], 'r-', **kwargs)
    
    elif rgn['shape'] == 'circle':
        patch = mpatches.Circle((rgn['params']['cx'], rgn['params']['cy']), 
                                rgn['params']['r'], alpha=alpha, transform=ax.transData)
        #plt.figure().artists.append(patch)
        ax.add_patch(patch)
        
    elif rgn['shape'] == 'polygon':
        for poly in rgn['params']['Polygons']:
            patch = mpatches.Polygon(poly.get_vertices(), closed=True, 
                                     alpha=alpha, transform=ax.transData)
        ax.add_patch(patch)
        
    elif rgn['shape'] == 'pixel':
        ax.plot(region['params']['cy'], region['params']['cx'], 'rs', ms=5)

def split_str(str):
    """
    Splits text from digits in a string.
    """
    
    logger = logging.getLogger(__name__)
    
    logger.debug('{0}'.format(str))
    
    match = re.match(r"([0-9]+.?\d{0,32}?)(d|m|s)", str)
    
    if match:
        items = match.groups()
            
    return items[0], items[1]
    
def main(out, cube, region, mode, show_region, plot_spec, faxis, stokes):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    hdulist = fits.open(cube)
    head = hdulist[0].header
    data = np.ma.masked_invalid(hdulist[0].data)
    data.fill_value = np.nan
    
    naxis = head['NAXIS']
    logger.debug('NAXIS: {0}'.format(naxis))
    
    if stokes:
        logger.info("Will now swap axis {0} with {1}".format(0, 1))
        logger.info("to leave stokes as last axes.")
        data = np.swapaxes(data, 0, 1)
        logger.info("New data shape: {0}.".format(data.shape))
    
    # Build a WCS object to handle sky coordinates
    if not 'pix' in region:
        wcs = set_wcs(head)
    else:
        wcs = None
    
    # Only pass spatial axes
    if not 'pix' in region:
        if wcs.naxis > 2:
            rgn = parse_region(region, wcs.dropaxis(2))
    else:
        rgn = parse_region(region, wcs)
    
    # Add beam area info to the region. Used when the requested units are flux units.
    if 'flux' in mode.lower():
        rgn['barea'] = ci.beam_area_pix(head)
        logger.debug('Conversion to flux: {0}'.format(rgn['barea']))
    
    # Get the frequency axis
    if faxis <= 4 and len(data.shape) > 2:
        freq = get_axis(head, faxis)
        freq = np.ma.masked_invalid(freq)
        logger.debug('Invalid values will be replaced with: {0}'.format(freq.fill_value))
    else:
        fcol = [s for s in head.keys() if "FREQ" in s]
        try:
            freq = head[fcol[0]]
            logger.debug('No frequency axis. Will use column {0} of header.'.format(fcol))
        except IndexError:
            freq = np.nan
            logger.debug('Could not guess the frequency. Will use {0} as frequency.'.format(freq))
        
    #freq.fill_value = np.nan
    
    logger.info("Will now extract the spectra,",)
    logger.info("using mode: {0}.".format(mode))
    
    spec = extract_spec(data, rgn, naxis, mode)
    try:
        spec.fill_value = np.nan
        logger.info("Extracted spec has shape: {0}.".format(spec.shape))
        fspec = spec.filled()
    except AttributeError:
        logger.info("Extracted spec has shape: {0}.".format(1))
        fspec = spec
        
    # Try to guess the units of each output column based on the header.
    try:
        funit = head['CUNIT{0}'.format(int(faxis))]
    except KeyError:
        funit = '?'
    
    try:
        ftype = head['CTYPE{0}'.format(int(faxis))]
    except KeyError:
        ftype = '?'
    
    try:
        bunit = head['BUNIT']
    except KeyError:
        bunit = '?'
    
    if 'sum' in mode.lower():
        bunit = 'sum of {0}'.format(bunit)
    elif 'flux' in mode.lower():
        bunit = bunit.split('/')[0]
    
    logger.debug('Brightness unit: {0}'.format(bunit))
    logger.debug('Frequency unit: {0}'.format(funit))
    logger.debug('ftype: {0}'.format(ftype))
    try:
        logger.debug('len freq: {0}'.format(len(freq)))
    except TypeError:
        logger.debug('len freq: {0}'.format(1))
    try:    
        logger.debug('len spec: {0}'.format(len(fspec)))
    except TypeError:
        logger.debug('len spec: {0}'.format(1))
    
    tbtable = Table(np.array([freq, fspec]).T, 
                    names=['{0} {1}'.format(ftype, funit),
                           'Tb {0}'.format(bunit)],
                    dtype=[np.float64, np.float64])

    logger.debug('Writing output spectrum to: {0}'.format(out))
    ascii.write(tbtable, out, format='commented_header', overwrite=True)
    
    if show_region:
        logger.debug('Plotting extraction region.')
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1, 1, 1)
        if naxis == 2:
            ax.imshow(data, interpolation='none', origin='lower')
        else:
            try:
                ax.imshow(data.sum(axis=0).sum(axis=0), origin='lower', interpolation='none')
            except TypeError:
                ax.imshow(data.sum(axis=0), interpolation='none', origin='lower')
        show_rgn(ax, rgn)
        #ci.draw_beam(head, ax) # This requires a pywcsgrid2 object to work
        ax.autoscale(False)
        
        plt.savefig('{0}_extract_region_{1}.png'.format(cube, region), 
                    bbox_inches='tight', pad_inches=0.3)
    
    if plot_spec:
        plotspec(cube, freq, spec, ftype, funit, bunit, plot_spec)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('cube', type=str,
                        help="Cube to extract the spectrum from. (string)")
    parser.add_argument('region', type=str,
                        help="Spatial region where pixels are averaged. " \
                             "e.g., point,sky,18h46m22s,-2d56m12s")
    parser.add_argument('out', type=str,
                        help="Output file name.")
    parser.add_argument('-m', '--mode', type=str, default='sum',
                        help="Mode of extraction. Can be sum, avg or flux.")
    parser.add_argument('-r', '--show_region', action='store_true',
                        help='Plot the extraction region?')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose output?")
    parser.add_argument('-l', '--logfile', type=str, default=None,
                        help="Where to store the logs.\n" \
                             "(string, Default: output to console)")
    parser.add_argument('-p', '--plot_spec', type=str, default=None,
                        help="Plot the extracted spectrum to the given file.")
    parser.add_argument('-f', '--faxis', type=int, default=3,
                        help="Axis to use as spectral axis.")
    parser.add_argument('-s', '--stokes_last', action='store_true',
                        help="Stokes axis is last?")
    args = parser.parse_args()
            
    if args.verbose:
        loglev = logging.DEBUG
    else:
        loglev = logging.ERROR
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=args.logfile, level=loglev, format=formatter)
    
    logger = logging.getLogger(__name__)
    logger.info('Will extract a spectrum from cube: {0}'.format(args.cube))
    logger.info('Will extract region: {0}'.format(args.region))
    
    main(args.out, args.cube, args.region, args.mode, args.show_region, args.plot_spec, args.faxis, args.stokes_last)
    
