#!/usr/bin/env python

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
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
startTime = datetime.now()

def bpcorr(data, bandpass, head):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    hdu = fits.open(bandpass)
    bp = np.ma.masked_invalid(hdu[0].data) + 10.
    hbp = hdu[0].header
    
    if head != hbp:
        logger.info('Headers do not match, will interpolate the bandpass solutions.')
        ra, de, ve = ci.get_fits3axes(head)
        rs, ds, vs = ci.check_ascending(ra, de, ve, True)
        
        ibp = RegularGridInterpolator((ve[::vs], de[::ds], ra[::rs]), 
                                      bp[::vs,::ds,::rs].filled(), 
                                      bounds_error=False, fill_value=10.)
        
        bp4c = np.ones(data.shape)*10.
        for k in range(len(ve)):
            pts = np.array([[ve[k], de[j], ra[i]] for j in range(len(de)) for i in range(len(ra))])
            newshape = (1,) + data.shape[1:]
            bp4c[k] += ibp(pts).reshape(newshape)[0]
    else:
        bp4c = bp
        
    return ((data + 10.)/bp4c - 1.)*10.
    
    # Make a cube with the bandpass solutions
    #cbp = np.reshape(np.array([xbp]*np.prod(np.array(data.shape[1:]))).T, data.shape, order='C')

def solve(x, data, bandpass, cell, head, order):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    logger.info('Will solve for the bandpass on {0} pixels'.format(cell))
    
    nx = data.shape[2]/cell[0]
    cx = cell[0]
    ny = nx
    cy = cx
    if len(cell) > 1:
        ny = data.shape[1]/cell[1]
        cy = cell[1]
    
    bp_cube = np.ones(data.shape)*10.
 
    for i in range(nx):
        for j in range(ny):
            
            logger.info('{0:.0f}%'.format(i*j/(nx*ny)*100))
            y = data[:,j*cy:(j+1)*cy,i*cx:(i+1)*cx].mean(axis=1).mean(axis=1)

            # Turn NaNs to zeros
            my = np.ma.masked_invalid(y)
            mx = np.ma.masked_where(np.ma.getmask(my), x)
            mmx = np.ma.masked_invalid(mx)
            mmy = np.ma.masked_where(np.ma.getmask(mmx), my)
            np.ma.set_fill_value(mmy, 10.)
            np.ma.set_fill_value(mmx, 10.)
            gx = mmx.compressed()
            gy = mmy.compressed()
            
            bp = np.polynomial.polynomial.polyfit(gx, gy, order)
            # Interpolate and extrapolate to the original x axis
            xbp = np.polynomial.polynomial.polyval(x, bp)
            shape = bp_cube[:,j*cy:(j+1)*cy,i*cx:(i+1)*cx].shape
            bp_cube[:,j*cy:(j+1)*cy,i*cx:(i+1)*cx] = np.reshape(np.array([xbp]*np.prod(np.array(shape[1:]))).T, shape, order='C')
            
    logger.info('Will write the bandpass cube with the solutions.')
    hdulist = fits.PrimaryHDU(bp_cube)
    hdulist.header = deepcopy(head)
    hdulist.writeto(bandpass)

def main(cube, output, bandpass, mode, cell, order):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    
    logger.info('Processing cube {0}'.format(cube))

    # Load the data
    hdu = fits.open(cube)
    head = hdu[0].header
    data = np.ma.masked_invalid(hdu[0].data)
    x =  crrls.get_axis(head, 3)
    
    if len(data.shape) > 3:
        logger.info('Will drop first axis.')
        data = data[0]
    
    if mode.lower() in ['solve', 'solve apply']:
        data_bpc = solve(x, data, bandpass, cell, head, order)
    if mode.lower() in ['apply', 'solve apply']:
        logger.info('Will apply the bandpass solutions.')
        data_bpc = bpcorr(data, bandpass, head)
        # Only save if applying solutions
        logger.info('Will write the bandpass corrected cube.')
        data_bpc.fill_value = np.nan
        hdulist = fits.PrimaryHDU(data_bpc.filled())
        # Copy the header from original cube
        hdulist.header = head.copy()
        # Write the fits file
        hdulist.writeto(output)

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
                             "['solve', 'apply', 'solve apply']\n" \
                             "Default: apply")
    parser.add_argument('-k', '--order', type=int, default=5,
                        help="Polynomial order used for solving (int).\n" \
                             "Default: 5")
    parser.add_argument('--cell', type=str, default='(10,10)',
                        help="Cell size in pixels units used for spatial averaging while solving (string).\n" \
                             "Default=(10,10)")
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
    
    main(args.cubes, args.output, args.bandpass, args.mode, cell, args.order)

    logger.info('Script run time: {0}'.format(datetime.now() - startTime))
