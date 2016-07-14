#!/usr/bin/env python
"""
TODO: Implement parallelization in velocity.
"""

import numpy as np
import glob
import sys
import argparse
import logging
from astropy.io import fits
from crrlpy import crrls
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
startTime = datetime.now()

def stack_cubes(cubes, outfits, vmax, vmin, dv, weight, weight_list=None, v_axis=3, clobber=False, algo='channel'):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    cubel = glob.glob(cubes)
    
    if dv == 0:
        logger.info('Velocity width not specified.'), 
        logger.info('Will try to determine from input data.')
        for i,cube in enumerate(cubel):
            hdu = fits.open(cube)
            head = hdu[0].header
            x = crrls.get_axis(head, v_axis)
            if i == 0:
                dv = crrls.get_min_sep(x)
                vmax_min = max(x)
                vmin_max = min(x)
            else:
                dv = max(dv, crrls.get_min_sep(x))
                vmax_min = min(vmax_min, max(x))
                vmin_max = max(vmin_max, min(x))
        logger.info('Will use a velocity width of {0}'.format(dv))
        
        # Check velocity ranges to avoid latter crashes.
        if vmax_min < vmax:
            logger.info('Requested maximum velocity is larger '\
                        'than one of the cubes velocity axis.')
            logger.info('v_max={0}, v_max_min={1}'.format(vmax, vmax_min))
            logger.info('Will now exit')
            sys.exit(1)
        if vmin_max > vmin:
            logger.info('Requested minimum velocity is smaller '\
                        'than one of the cubes velocity axis.')
            logger.info('v_min={0}, v_min_min={1}'.format(vmin, vmin_min))
            logger.info('Will now exit')
            sys.exit(1)

    shape = hdu[0].shape
    if len(shape) > 3:
        logger.info('Will drop first axis.')
        s = 1
    else:
        s = 0
    
    nvaxis = np.arange(vmin, vmax, dv)
    stack = np.zeros((len(nvaxis),) + shape[s+1:])
    logger.info('Output stack will have dimensions: {0}'.format(stack.shape))
    
    for i,cube in enumerate(cubel):
        
        logger.info('Adding cube {0}/{1}'.format(i, len(cubel)-1))

        # Load the data
        hdu = fits.open(cube)
        head = hdu[0].header
        data = np.ma.masked_invalid(hdu[0].data)
        
        if len(data.shape) > 3:
            logger.info('Will drop first axis.')
            data = data[0]
        
        # Get the cube axes
        ra = crrls.get_axis(head, 1)
        de = crrls.get_axis(head, 2)
        ve = crrls.get_axis(head, v_axis)

        logger.info('RA axis limits: {0} {1}'.format(min(ra), max(ra)))
        logger.info('DEC axis limits: {0} {1}'.format(min(de), max(de)))
        logger.info('VEL axis limits: {0} {1}'.format(min(ve), max(ve)))
        
        # Check that the axes are in ascending order
        if ve[0] > ve[1]: 
            vs = -1
        else:
            vs = 1
            
        if ra[0] > ra[1]:
            vr = -1
        else:
            vr = 1
        
        # Interpolate the data
        interp = RegularGridInterpolator((ve[::vs], de, ra[::vr]), data[::vs,:,::vr])

        # Add the data to the stack
        if 'vector' in algo.lower():
            logger.info('Will use one vector to reconstruct the cube')
            pts = np.array([[nvaxis[k], de[j], ra[i]] \
                            for k in range(len(nvaxis)) \
                            for j in range(len(de)) \
                            for i in range(len(ra))])
            stack += interp(pts).reshape(stack.shape)

        elif 'channel' in algo.lower():
            logger.info('Will reconstruct the cube one channel at a time')
            for k in range(len(nvaxis)):
                pts = np.array([[nvaxis[k], de[j], ra[i]] \
                                for j in range(len(de)) \
                                for i in range(len(ra[::vr]))])
                newshape = (1,) + shape[s+1:]
                try:
                    stack[k] += interp(pts).reshape(newshape)[0]
                except ValueError:
                    logger.info('Point outside range: ' \
                                'vel={0}, ra={1}..{2}, dec={3}..{4}'.format(pts[0,0], 
                                                                            min(pts[:,1]), 
                                                                            max(pts[:,1]), 
                                                                            min(pts[:,2]), 
                                                                            max(pts[:,2])))
        else:
            logger.info('Cube reconstruction algorithm unrecognized.')
            logger.info('Will exit now.')
            sys.exit(1)

    # Divide by the number of input cubes to get the mean
    stack = stack/len(cubel)
    
    # Write to a fits file
    hdulist = fits.PrimaryHDU(stack)
    # Copy the header from the first channel image
    hdulist.header = fits.open(cubel[0])[0].header.copy()
    hdulist.header['CTYPE3'] = 'VELO'
    hdulist.header['CRVAL3'] = nvaxis[0]
    hdulist.header['CDELT3'] = dv
    hdulist.header['CRPIX3'] = 1
    hdulist.header['CUNIT3'] = 'm/s'
    hdulist.writeto(outfits, clobber=clobber)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('cubes', type=str,
                        help="List of cubes to process.\n" \
                             r"E.g., \"lba_hgh_*.ascii\" (string).\n" \
                             "Wildcards and [] accepted.")
    parser.add_argument('stack', type=str,
                        help="Output stack filename (string).")
    parser.add_argument('--weight', type=str, default=None,
                        help="Weight type used to stack.\n" \
                             "Can be None, sigma, sigma2 or list.\n" \
                             "None uses no weight.\n" \
                             "sigma uses the inverse of the spectra rms.\n" \
                             "sigma2 uses the square of the inverse of \n" \
                             "the spectra rms.\n" \
                             "list uses a user provided list.\n" \
                             "The list should have 2 columns, the first with\n" \
                             "the spectrum filename and the second with the weight.\n" \
                             "E.g., lba_hgh_n455 0.2\n" \
                             "      lba_hgh_n456 0.8\n" \
                             "      lba_hgh_n457 0.5\n" \
                             "      ...          ...")
    parser.add_argument('--weight_list', type=str, default=None,
                        help="File with list of spectrum and their weights.\n" \
                             "Default: None")
    parser.add_argument('--v_max', type=float, default=None, required=True,
                        help="Maximum velocity to include in stack.\n" \
                             "Default: None")
    parser.add_argument('--v_min', type=float, default=None, required=True,
                        help="Minimum velocity to include in stack.\n" \
                             "Default: None")
    parser.add_argument('-d', '--dv', type=float, default=0,
                        help="Velocity resolution of stack.\n" \
                             "If 0 will try to determine inside script.\n" \
                             "Default: 0")
    parser.add_argument('--v_axis', type=int, default=3,
                        help="Axis with the velocity information. Default: 3")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose output?")
    parser.add_argument('-l', '--logfile', type=str, default=None,
                        help="Where to store the logs.\n" \
                             "(string, Default: output to console)")
    parser.add_argument('--clobber', 
                        help="Overwrite existing fits files?",
                        action='store_true')
    parser.add_argument('--algo', type=str, default='channel',
                        help="Which algorithm should be used for reconstructing the cubes?\n" \
                             "Default: \"channel\". [vector|channel]")
    args = parser.parse_args()
    
    if args.verbose:
        loglev = logging.DEBUG
    else:
        loglev = logging.ERROR
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=args.logfile, level=loglev, format=formatter)
    
    logger = logging.getLogger(__name__)

    stack_cubes(args.cubes, args.stack, args.v_max, args.v_min, args.dv, 
                args.weight, args.weight_list, args.v_axis, args.clobber, args.algo)
    
    logger.info('Script run time: {0}'.format(datetime.now() - startTime))
