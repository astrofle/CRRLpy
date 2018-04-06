#!/usr/bin/env python
"""
TODO:   Check behavior when cubes have missing channels.
        Implement parallelization in velocity.
"""

import numpy as np
import glob
import sys
import argparse
import logging
from astropy.io import fits
from crrlpy import crrls
from crrlpy import utils
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
startTime = datetime.now()

def parse_cube_list(cubes):
    """
    """

    logger = logging.getLogger(__name__)
    
    if glob.glob(cubes):
        return glob.glob(cubes)

    else:
        if len(cubes.split(',')) > 1:
            return cubes.split(',')
        else:
            logger.info('Input cube list not understood.')

def stack_cubes(cubes, outfits, vmax, vmin, dv, weight_list=None, v_axis=3, overwrite=False, algo='channel'):
    """
    """
    
    logger = logging.getLogger(__name__)
    
    cubel = cubes #glob.glob(cubes)
    logger.debug(cubel)
    
    if dv == 0:
        logger.info('Velocity width not specified.'), 
        logger.info('Will try to determine from input data.')
        for i,cube in enumerate(cubel):
            hdu = fits.open(cube)
            head = hdu[0].header
            x = crrls.get_axis(head, v_axis)
            if i == 0:
                dv = utils.get_min_sep(x)
                vmax_min = max(x)
                vmin_max = min(x)
            else:
                dv = max(dv, utils.get_min_sep(x))
                vmax_min = min(vmax_min, max(x))
                vmin_max = max(vmin_max, min(x))
            logger.debug('Cube: {0}'.format(cube))
            logger.debug('Cube velocity limits: {0} {1} {2}'.format(min(x), max(x), utils.get_min_sep(x)))
        logger.info('Will use a velocity width of {0}'.format(dv))
        
        # Check velocity ranges to avoid latter crashes.
        if vmax_min < vmax:
            logger.info('Requested maximum velocity is larger '\
                        'than one of the cubes velocity axis.')
            #logger.info('Conflicting cube: {0}'.format(cube))
            logger.info('v_max={0}, v_max_min={1}'.format(vmax, vmax_min))
            logger.info('Will now exit')
            sys.exit(1)
        if vmin_max > vmin:
            logger.info('Requested minimum velocity is smaller '\
                        'than one of the cubes velocity axis.')
            #logger.info('Conflicting cube: {0}'.format(cube))
            logger.info('v_min={0}, v_min_max={1}'.format(vmin, vmin_max))
            logger.info('Will now exit')
            sys.exit(1)

    shape = hdu[0].shape
    if len(shape) > 3:
        logger.info('Will drop first axis.')
        s = 1
    else:
        s = 0
    
    nvaxis = np.arange(vmin, vmax, dv)
    stack = np.ma.zeros((len(nvaxis),) + shape[s+1:])
    covrg = np.zeros((len(nvaxis),) + shape[s+1:])

    # Check if there is a weight list
    if weight_list:
        try:
            wl = np.loadtxt(weight_list, dtype=[('fits', np.str_, 256), ('w', np.float64)])
        except ValueError:
            wl = np.loadtxt(weight_list, dtype=[('fits', np.str_, 256), ('w', np.str_, 256)])
    weight = np.ma.zeros((len(nvaxis),) + shape[s+1:])
        
    logger.info('Output stack will have dimensions: {0}'.format(stack.shape))
    
    for i,cube in enumerate(cubel):
        
        logger.info('Adding cube {0}/{1}'.format(i, len(cubel)-1))

        # Load the data
        hdu = fits.open(cube)
        head = hdu[0].header
        data = np.ma.masked_invalid(hdu[0].data)

        # Check the weight
        if weight_list:
            w = wl['w'][np.where(wl['fits'] == cube)]
        else:
            w = 1
        logger.info('Will use a weight of {0}.'.format(w))
        try:
            aux_weight = np.ones(stack.shape)*w
        except TypeError:
            w = np.ma.masked_invalid(fits.open(w[0])[0].data)
        #aux_weight = np.ones(shape[s+1:])*w
        
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
       
        vmin_idx = utils.best_match_indx(vmin, ve)
        vmax_idx = utils.best_match_indx(vmax, ve)
        [vmin_idx,vmax_idx] = sorted([vmin_idx,vmax_idx])
        vmin_idx -= 1
        vmax_idx += 1
        data_ = data[vmin_idx:vmax_idx]
        ve_ = ve[vmin_idx:vmax_idx]
 
        # Check that the axes are in ascending order
        if ve_[0] > ve_[1]: 
            vs = -1
        else:
            vs = 1
            
        if ra[0] > ra[1]:
            vr = -1
        else:
            vr = 1
        
        # Interpolate the data
        interp = RegularGridInterpolator((ve_[::vs], de, ra[::vr]), data_[::vs,:,::vr])

        # Add the data to the stack
        if 'vector' in algo.lower():

            logger.info('Will use one vector to reconstruct the cube')
            pts = np.array([[nvaxis[k], de[j], ra[i]] \
                            for k in range(len(nvaxis)) \
                            for j in range(len(de)) \
                            for i in range(len(ra))])
            aux_weight = np.ones(stack.shape)*w
            stack += interp(pts).reshape(stack.shape)*aux_weight

        elif 'channel' in algo.lower():

            logger.info('Will reconstruct the cube one channel at a time')

            for k in range(len(nvaxis)):

                pts = np.array([[nvaxis[k], de[j], ra[i]] \
                                for j in range(len(de)) \
                                for i in range(len(ra[::vr]))])
                newshape = shape[s+1:]
                aux_weight = np.ones(newshape)*w
                aux_cov = np.ones(newshape)

                try:
                    stack_ = np.ma.masked_invalid(interp(pts).reshape(newshape)*aux_weight[0])
                    stack_.fill_value = 0.
                    aux_weight[stack_.mask] = 0
                    aux_cov[stack_.mask] = 0
                    weight[k] += aux_weight
                    covrg[k] += aux_cov
                    stack[k] += stack_.filled()
                    
                except ValueError:
                    logger.info('Point outside range: ' \
                                'vel={0}, ra={1}..{2}, dec={3}..{4}'.format(pts[0,0], 
                                                                            min(pts[:,2]), 
                                                                            max(pts[:,2]), 
                                                                            min(pts[:,1]), 
                                                                            max(pts[:,1])))
        
        else:
            logger.info('Cube reconstruction algorithm unrecognized.')
            logger.info('Will exit now.')
            sys.exit(1)
        
    # Divide by the number of input cubes to get the mean
    stack = stack/weight
    stack.fill_value = np.nan
    weight.fill_value = np.nan

    # Write to a fits file
    hdulist = fits.PrimaryHDU(stack.filled())
    # Copy the header from the first channel image
    hdulist.header = fits.open(cubel[0])[0].header.copy()
    hdulist.header['CTYPE3'] = 'VELO'
    hdulist.header['CRVAL3'] = nvaxis[0]
    hdulist.header['CDELT3'] = dv
    hdulist.header['CRPIX3'] = 1
    hdulist.header['CUNIT3'] = 'm/s'
    hdulist.writeto(outfits, overwrite=overwrite)

    stack_head = hdulist.header.copy()
    hdulist = fits.PrimaryHDU(weight.filled())
    hdulist.header = stack_head
    hdulist.header['CTYPE3'] = 'VELO'
    hdulist.header['CRVAL3'] = nvaxis[0]
    hdulist.header['CDELT3'] = dv
    hdulist.header['CRPIX3'] = 1
    hdulist.header['CUNIT3'] = 'm/s'
    hdulist.writeto(outfits.split('.fits')[0]+'_weight.fits', overwrite=overwrite)

    hdulist = fits.PrimaryHDU(covrg)
    hdulist.header = stack_head
    hdulist.header['CTYPE3'] = 'VELO'
    hdulist.header['CRVAL3'] = nvaxis[0]
    hdulist.header['CDELT3'] = dv
    hdulist.header['CRPIX3'] = 1
    hdulist.header['CUNIT3'] = 'm/s'
    hdulist.writeto(outfits.split('.fits')[0]+'_coverage.fits', overwrite=overwrite)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('cubes', type=str, nargs='+',
                        help="List of cubes to process.\n" \
                             r"E.g., \"lba_hgh_*.ascii\" (string).\n" \
                             "Wildcards and [] accepted.")
    parser.add_argument('stack', type=str,
                        help="Output stack filename (string).")
    parser.add_argument('--weight_list', type=str, default=None,
                        help="File with list of cubes and their weights.\n" \
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
    parser.add_argument('--overwrite', 
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

    #cubel = parse_cube_list(args.cubes)

    stack_cubes(args.cubes, args.stack, args.v_max, args.v_min, args.dv, 
                args.weight_list, args.v_axis, args.overwrite, args.algo)
    
    logger.info('Script run time: {0}'.format(datetime.now() - startTime))
