#!/usr/bin/env python
"""
Convert a cube to optical depth following Equation (1) of Salas et al. (2017, doi:10.1093/mnras/stx239).
Handles one cube at a time. The continuum image and line cube should have the same WCS and spatial size. 
The continuum image will be expanded into a cube which is used to subtract from the line cube.
The output cube will have the same header as the input cube, except for the BUNIT keyword.
"""

from __future__ import division

import sys
import logging
import argparse

import numpy as np
import astropy.io.fits as fits

from datetime import datetime
startTime = datetime.now()

def main(cube, cont, outp, overwrite=False):
    """
    Main body of the script: 2tau.py.
    Converts a cube to optical depth units.
    The output cube will have the same header as the input cube.
    """
    
    hdu = fits.open(cube)
    cube = np.ma.masked_invalid(hdu[0].data)
    head = hdu[0].header
    
    hdu = fits.open(cont)
    cont = np.ma.masked_invalid(hdu[0].data)
    
    if len(cube.shape) > 3:
        logger.info('Will drop Stokes axis from the cube.')
        cube = cube[0]
        
    if len(cont.shape) > 3:
        logger.info('Will drop Stokes axis from the continuum.')
        cont = cont[0]

    if len(cont.shape) > 2:
        logger.info('Will drop spectral axis from the continuum.')
        cont = cont[0]
    
    logger.debug('cube shape: {0}'.format(cube.shape))
    logger.debug('cont shape: {0}'.format(cont.shape))    
    
    econt = np.ma.masked_invalid(([cont]*cube.shape[0])).reshape(cube.shape)
    
    csub = np.ma.divide(cube, econt) - 1.
    
    save(csub, outp, head, overwrite=overwrite)

def parse_args():
    """
    """
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('cube', type=str,
                        help="Cube with spectral information to process.\n" \
                             "E.g., lba_hgh_SB120.fits (string).\n")
    parser.add_argument('cont', type=str,
                        help="Image with continuum.\n" \
                             "E.g., lba_hgh_SB120_cont.fits (string).\n")
    parser.add_argument('output', type=str,
                        help="Output cube filename (string).")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose output?")
    parser.add_argument('-l', '--logfile', type=str, default=None,
                        help="Where to store the logs.\n" \
                             "(string, Default: output to console)")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing cubes?.\n" \
                             "Default: False")
    
    args = parser.parse_args()
    
    return args

def save(data, output, head, overwrite=False):
    """
    Save a fits file.
    """
    
    data.fill_value = np.nan
    hdulist = fits.PrimaryHDU(data.filled())
    # Copy the header from original cube
    hdulist.header = head.copy()
    hdulist.header['BUNIT'] = 'TAU'
    # Write the fits file
    hdulist.writeto(output, overwrite=overwrite)

if __name__ == '__main__':
    
    args = parse_args()
    
    # Set up logger
    if args.verbose:
        loglev = logging.DEBUG
    else:
        loglev = logging.ERROR
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=args.logfile, level=loglev, format=formatter)
    
    logger = logging.getLogger(__name__)
    
    main(args.cube, args.cont, args.output, overwrite=args.overwrite)
    
    logger.info('Script run time: {0}'.format(datetime.now() - startTime))
