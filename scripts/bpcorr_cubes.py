#!/usr/bin/env python
"""
Bandpass correction script for cubes.
It provides tools for deriving bandpass 
solutions, masking, applying and smoothing.

TODO: normalize convolution.
"""
from __future__ import division

import sys
import glob
import logging
import argparse
from copy import deepcopy
import numpy as np
import scipy.interpolate as si
import crrlpy.crrls as crrls
import crrlpy.imtools as ci
import astropy.io.fits as fits
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
startTime = datetime.now()

def bpcorr(data, bandpass, head):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    hdu = fits.open(bandpass)
    bp = np.ma.masked_invalid(hdu[0].data)
    hbp = hdu[0].header
    
    if len(bp.shape) > 3:
        logger.info('Will drop first axis from bandpass solution.')
        bp = bp[0]
    
    if not ci.compare_headers(hbp, head):
        logger.info('Headers do not match, will interpolate the bandpass solutions.')
        ra, de, ve = ci.get_fits3axes(head)
        rabp, debp, vebp = ci.get_fits3axes(hbp)
        rs, ds, vs = ci.check_ascending(rabp, debp, vebp, True)
        
        ibp = RegularGridInterpolator((vebp[::vs], debp[::ds], rabp[::rs]), 
                                      bp[::vs,::ds,::rs].filled(), 
                                      bounds_error=False, fill_value=10.)
        
        bp4c = np.zeros(data.shape)
        for k in range(len(ve)):
            pts = np.array([[ve[k], de[j], ra[i]] for j in range(len(de)) for i in range(len(ra))])
            newshape = (1,) + data.shape[1:]
            bp4c[k] += ibp(pts).reshape(newshape)[0]
    else:
        bp4c = bp
        
    return ((data)/bp4c - 1.)*10.

def interpolate_bpsol(x, bp, head):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    ra, de, ve = ci.get_fits3axes(head)
    rs, ds, vs = ci.check_ascending(ra, de, x, True)
    
    ibp = RegularGridInterpolator((x[::vs], de[::ds], ra[::rs]), 
                                  bp[::vs,::ds,::rs], 
                                  bounds_error=False, fill_value=10.)
    
    bp4c = np.zeros((len(ve), len(de), len(ra)))
    for k in range(len(ve)):
        pts = np.array([[ve[k], de[j], ra[i]] for j in range(len(de)) for i in range(len(ra))])
        newshape = (1, len(de), len(ra))
        bp4c[k] += ibp(pts).reshape(newshape)[0]
        
    return np.ma.masked_equal(bp4c, 0.0)

def mask_cube(vel, data, vel_rngs):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    vel_indx = np.empty(vel_rngs.shape, dtype=int)
    nvel_indx = np.empty(vel_rngs.shape, dtype=int)
    chns = np.zeros(len(vel_rngs))
    nchns = np.zeros(len(vel_rngs))
    nvel = []
    extend = nvel.extend

    for i,velrng in enumerate(vel_rngs):

        vel_indx[i][0] = crrls.best_match_indx(velrng[0], vel)
        vel_indx[i][1] = crrls.best_match_indx(velrng[1], vel)
        
        nvel.extend(vel[vel_indx[i][0]:vel_indx[i][1]+1])
        
        nvel_indx[i][0] = crrls.best_match_indx(velrng[0], nvel)
        nvel_indx[i][1] = crrls.best_match_indx(velrng[1], nvel)
        
        chns[i] = vel_indx[i][1] - vel_indx[i][0] + 1
        nchns[i] = nvel_indx[i][1] - nvel_indx[i][0] + 1
    
    mdata = np.ones(((int(sum(chns)),)+data.shape[1:]))*10.
    logger.info('Masked data shape: {0}'.format(mdata.shape))

    logger.info('Will select the unmasked data.')
    for i in range(len(vel_indx)):
        mdata[nvel_indx[i][0]:nvel_indx[i][1]+1] = data[vel_indx[i][0]:vel_indx[i][1]+1]
        
    return nvel, mdata

def save(data, output, head, clobber=False):
    """
    """
    
    data.fill_value = np.nan
    hdulist = fits.PrimaryHDU(data.filled())
    # Copy the header from original cube
    hdulist.header = head.copy()
    # Write the fits file
    hdulist.writeto(output, clobber=clobber)

def smooth(bp_cube, std):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    gauss_kernel = Gaussian2DKernel(std)
    bp_cube_sm = np.ma.masked_invalid(np.empty(bp_cube.shape))
    for v in range(bp_cube.shape[0]):
        logger.info('{0:.0f}%'.format(v/bp_cube.shape[0]*100.))
        bp_cube_sm[v] = np.ma.masked_invalid(convolve(bp_cube[v], gauss_kernel))
        
    return bp_cube_sm

def solve(x, data, bandpass, cell, head, order, oversample=1):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    logger.info('Will solve for the bandpass on {0} pixels'.format(cell))
    
    cx = cell[0]//oversample
    nx = (data.shape[2]//cx)
    ny = nx
    cy = cx
    if len(cell) > 1:
        cy = cell[1]//oversample
        ny = (data.shape[1]//cy)
    
    bp_cube = np.ma.masked_invalid(np.zeros(data.shape))
    bp_cube_cov = np.ma.masked_invalid(np.zeros(data.shape))
    
    for i in range(nx):
        logger.info('{0:.0f}%'.format(i/nx*100.))
        for j in range(ny):
            y0 = j*cy
            yf = (j+1)*cy
            x0 = i*cx
            xf = (i+1)*cx
            #print j, ny
            #print y0, yf, x0, xf
            y = np.ma.masked_invalid(data[:,y0:yf,x0:xf].mean(axis=1).mean(axis=1))

            # Turn NaNs to zeros
            my = np.ma.masked_invalid(y)
            mx = np.ma.masked_where(np.ma.getmask(my), x)
            mmx = np.ma.masked_invalid(mx)
            mmy = np.ma.masked_where(np.ma.getmask(mmx), my)
            np.ma.set_fill_value(mmy, 10.)
            np.ma.set_fill_value(mmx, 10.)
            gx = mmx.compressed()
            gy = mmy.compressed()
            
            # Derive a polynomial to correct the bandpass
            bp = np.polynomial.polynomial.polyfit(gx, gy, order)
            
            # Interpolate and extrapolate to the original x axis
            xbp = np.polynomial.polynomial.polyval(x, bp)
            shape = bp_cube[:,y0:yf,x0:xf].shape
            bp_cube[:,y0:yf,x0:xf] += np.reshape(np.array([xbp]*np.prod(np.array(shape[1:]))).T, 
                                                 shape, order='C')
            bp_cube_cov[:,y0:yf,x0:xf] += np.ones(shape)
            
    logger.info('Bandpass solution spatial pixels: {0:.0f}/{1:.0f} '\
                '{2:.0f}/{3:.0f}'.format((i+1)*cx, data.shape[2], (j+1)*cy, data.shape[1]))

    return np.ma.divide(bp_cube, bp_cube_cov)

def main(cube, output, bandpass, mode, cell, order, std=11, vrngs=None, oversample=1, average=1):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    
    logger.info('Processing cube {0}'.format(cube))

    # Load the data
    hdu = fits.open(cube)
    head = hdu[0].header
    data = np.ma.masked_invalid(hdu[0].data) + 10.
    x = crrls.get_axis(head, 3)
    
    logger.info('Data shape: {0}'.format(data.shape))
    
    # Remove Stokes axis if present
    if len(data.shape) > 3:
        logger.info('Will drop first axis.')
        data = data[0]
    
    # Average
    if average > 1:
        logger.info('Will average {0} channels together.'.format(average))
        avg_x = crrls.average(x, 0, average)
        avg_data = crrls.average(data, 0, average)
        logger.info('Averaged data shape: {0}'.format(avg_data.shape))
    else:
        avg_x = x
        avg_data = data
    # Mask
    if mode.lower() in ['mask', 'mask solve', 'mask solve apply', 'mask solve smooth apply']:
        logger.info('Will mask the requested velocity ranges before solving.')
        mx, mdata = mask_cube(avg_x, avg_data, vrngs)
    else:
        mx = avg_x
        mdata = avg_data
    # Solve
    if mode.lower() in ['solve', 'mask solve', 'mask solve apply', 'solve apply', 'solve smooth apply', 'mask solve smooth apply']:
        bp_cube = solve(mx, mdata, bandpass, cell, head, order, oversample)
    if not ('smooth' in mode.lower() or 'mask' in mode.lower()) and 'solve' in mode.lower() and average <= 1:
        logger.info('Will write the bandpass cube with the solutions.')
        save(bp_cube, bandpass, head)
    if 'mask solve' in mode.lower() or average > 1:
        logger.info('Will write the bandpass cube with the solutions interpolated to the original axis.')
        bp_cube = interpolate_bpsol(mx, bp_cube, head)
        save(bp_cube, bandpass, head)
    # Smooth
    if mode.lower() in ['mask solve smooth', 'solve smooth', 'solve smooth apply', 'mask solve smooth apply']:
        logger.info('Will smooth the bandpass solution with a ' \
                    'Gaussian kernel of standard deviation: {0}.'.format(std))
        bp_cube = smooth(bp_cube, std)
        logger.info('Will write the smoothed bandpass cube with the solutions.')
        save(bp_cube, bandpass, head)
    # Apply with mask
    if 'apply' in mode.lower():
        logger.info('Will apply the bandpass solutions.')
        data_bpc = bpcorr(data, bandpass, head)
        # Only save if applying solutions
        logger.info('Will write the bandpass corrected cube.')
        save(data_bpc, output, head)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('cubes', type=str,
                        help="Cube to process.\n" \
                             "E.g., lba_hgh_SB120.fits (string).\n")
    parser.add_argument('output', type=str,
                        help="Output cube filename (string).")
    parser.add_argument('bandpass', type=str,
                        help="Bandpass to be applied (string).")
    parser.add_argument('-m', '--mode', type=str, default='apply',
                        help="Mode of operation (string).\n" \
                             "['mask', 'solve', 'smooth', 'apply', 'solve apply',\n" \
                              "'solve smooth apply', 'mask solve smooth apply']\n" \
                             "Default: 'apply'")
    parser.add_argument('--vrngs', type=str, default=None,
                        help="Velocity ranges to keep while solving for the bandpass (string).\n" \
                             "Default: None")
    parser.add_argument('-k', '--order', type=int, default=5,
                        help="Polynomial order used for solving (int).\n" \
                             "Default: 5")
    parser.add_argument('--cell', type=str, default='(10,10)',
                        help="Cell size in pixels units used for spatial averaging while solving (string).\n" \
                             "Default=(10,10)")
    parser.add_argument('--std', type=int, default=11,
                        help="Standard deviation of the Gaussian kernel (float).\n" \
                             "Default: 11")
    parser.add_argument('--oversample', type=int, default=1,
                        help="Derive bandpass solutions (float).\n" \
                             "Default: 11")
    parser.add_argument('-a', '--average', type=int, default=1,
                        help="Average the velocity axis by this amount (int).\n" \
                             "Default: 1")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose output?")
    parser.add_argument('-l', '--logfile', type=str, default=None,
                        help="Where to store the logs.\n" \
                             "(string, Default: output to console)")
    args = parser.parse_args()
    
    if args.verbose:
        loglev = logging.DEBUG
    else:
        loglev = logging.ERROR
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=args.logfile, level=loglev, format=formatter)
    
    logger = logging.getLogger(__name__)
    
    strcell = args.cell[1:-1].split(',')
    cell = [int(strcell[0])]
    if len(strcell) > 1:
        cell.append(int(strcell[1]))
    
    if args.vrngs:
        vrngss = args.vrngs[1:-1].split(',')
        vrngs = np.empty((len(vrngss)//2,2))
        for i in range(0,len(vrngss),2):
            vrngs[i//2][0] = float(vrngss[i])
            vrngs[i//2][1] = float(vrngss[i+1])
    else:
        vrngs = None
        
    main(args.cubes, args.output, args.bandpass, args.mode, cell, args.order, 
         args.std, vrngs, args.oversample, args.average)

    logger.info('Script run time: {0}'.format(datetime.now() - startTime))
