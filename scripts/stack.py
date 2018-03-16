#!/usr/bin/env python
"""
Simple stacking script.
Works on 1D spectra.
"""

import sys
import glob
import logging
import argparse
import numpy as np

from scipy import interpolate

from crrlpy import crrls
from crrlpy import utils

def stack_interpol(specs, output, vmax, vmin, dv, x_col, y_col, weight, weight_list=None, weight_list_cols='0,1'):
    """
    """
    
    #specs = glob.glob(spec)
    
    # If only one file is passed, it probably contains the list
    #if len(specs) == 1:
        #specs = np.genfromtxt(specs[0], dtype=str)
        
    # Setup weight list columns, if need be.
    if weight_list:
        wlc0,wlc1 = list(map(int, weight_list_cols.split(',')))
        print(wlc0,wlc1)
    
    logger.info('Will use the following files as input: ')
    logger.info(specs)
    
    # Determine velocity resolution of stack.
    if dv == 0:
        logger.info('Will determine the velocity resolution of the stack.')
        for i,s in enumerate(specs):
            data = np.loadtxt(s)
            x = data[:,x_col]
            if i == 0:
                dv = utils.get_min_sep(x)
            else:
                dv = max(dv, utils.get_min_sep(x))
            logger.info('Spectrum {0} has velocity resolution of: {1}'.format(s, utils.get_min_sep(x)))
    logger.info('Stack velocity resolution: {0}'.format(dv))

    # Setup the grid for the stack.
    xgrid = np.arange(vmin, vmax, dv)
    ygrid = np.zeros(len(xgrid))      # the temperatures
    zgrid = np.zeros(len(xgrid))      # the number of tb points in every stacked channel
    
    for i,s in enumerate(specs):
        
        logger.info('Working on file: {0}'.format(s))
        
        data = np.loadtxt(s)
        x = data[:,x_col]
        y = data[:,y_col]
        
        # Sort by velocity
        o = x.argsort()
        x = x[o]
        y = y[o]
        
        # Catch NaNs and invalid values:
        mask_x = np.ma.masked_equal(x, -9999).mask
        mask_y = np.isnan(y)
        mask = np.array(reduce(np.logical_or, [mask_x, mask_y]))
        
        # Interpolate non masked ranges indepently.
        my = np.ma.masked_where(mask, y)
        mx = np.ma.masked_where(mask, x)
        valid = np.ma.flatnotmasked_contiguous(mx)
        y_aux = np.zeros(len(xgrid))
        if not isinstance(valid, slice):
            for j,rng in enumerate(valid):
                #print "slice {0}: {1}".format(i, rng)
                if len(x[rng]) > 1:
                    interp_y = interpolate.interp1d(x[rng], y[rng],
                                                    kind='linear',
                                                    bounds_error=False,
                                                    fill_value=0.0)
                    y_aux += interp_y(xgrid)
                elif not np.isnan(x[rng]):
                    #print x[rng]
                    y_aux[utils.best_match_indx(x[rng], xgrid)] += y[rng]                    
        else:
            #print "slice: {0}".format(valid)
            interp_y = interpolate.interp1d(x[valid], y[valid],
                                            kind='linear',
                                            bounds_error=False,
                                            fill_value=0.0)
            y_aux += interp_y(xgrid)
        
        # Check which channels have data.
        ychan = [1 if ch != 0 else 0 for ch in y_aux]
        
        # Determine the weight.
        if not weight:
            w = np.ones(len(xgrid))
            
        elif weight == 'list':
            wl = np.loadtxt(weight_list, dtype=str)
            w = float(wl[:,wlc1][np.where(wl[:,wlc0] == s)[0][0]])
            
        elif weight == 'sigma':
            w = 1./crrls.get_rms(my)
            
        elif weight == 'sigma2':
            w = 1./np.power(crrls.get_rms(my), 2)
        
        logger.info('Will use a weight of: {0}'.format(w))
        
        # Stack!
        zgrid = zgrid + np.multiply(ychan, w)
        ygrid = ygrid + y_aux*np.multiply(ychan, w)
            
    # Divide by the total weight to preserve optical depth
    ygrid = np.divide(ygrid, zgrid)
        
    np.savetxt(output, np.c_[xgrid, ygrid, zgrid], header="x axis, " \
                                                          "stacked y axis, " \
                                                          "y axis weight")
    
def stack_filter(specs, output, vmax, vmin, dv, x_col, y_col, window, window_opts):
    """
    """
    
    xgrid = []
    ygrid = []
    zgrid = []
    
    #specs = glob.glob(spec)

    # If only one file is passed, it probably contains the list
    #if len(specs) == 1:
        #specs = np.genfromtxt(specs[0], dtype=str)

    #print(specs)

    if dv == 0:
        for s in specs:
            data = np.loadtxt(s)
            x = data[:,x_col]
            dv = utils.get_min_sep(x)
            dv = max(dv, utils.get_min_sep(x))
    # Loop over the spectra to stack
    for i,s in enumerate(specs):
        data = np.loadtxt(s)
        x = data[:,x_col] 
        y = data[:,y_col] 
        
        # Catch NaNs and invalid values:
        mask_x = np.ma.masked_equal(x, 1.0).mask
        mask_y = np.isnan(y)
        mask = np.array(reduce(np.logical_or, [mask_x, mask_y]))
        
        # Remove NaNs and invalid values
        x = x[~mask]
        y = y[~mask]
        
        # Sort by velocity
        #x, y = (list(xy) for xy in zip(*sorted(zip(x, y))))
        
        # Append
        xgrid.append(x)
        ygrid.append(y)
        
        # Determine the spectral resolution
        if dv == 0:
            dv = utils.get_min_sep(x)
            dv_max = max(dv, utils.get_min_sep(x))
            dv_min = min(dv, utils.get_min_sep(x))
            dv = 0
        
    # Flatten the stacked spectra
    #xgrid = list(itertools.chain.from_iterable(xgrid))
    xgrid = np.array(list(_x for _xg in xgrid for _x in _xg))
    #ygrid = list(itertools.chain.from_iterable(ygrid))
    ygrid = np.array(list(_y for _yg in ygrid for _y in _yg))
    
     # Sort by velocity
    xgrid, ygrid = (list(xy) for xy in zip(*sorted(zip(xgrid, ygrid))))
    
    # Apply a window function to reduce noise
    if window == 'Wiener':
        zgrid = crrls.Wiener(ygrid, **window_opts)
    elif window == 'SavGol':
        zgrid = crrls.SavGol(ygrid, **window_opts)
    elif window == 'Gauss':
        zgrid = crrls.Gauss(ygrid, **window_opts)
        
    if dv == 0:
        dv = np.mean([dv_min,dv_max])
    reg_xgrid = np.arange(vmin, vmax, dv)
    interp_y = interpolate.interp1d(xgrid, ygrid,
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=0.0)
    ygrid = interp_y(reg_xgrid)
    interp_z = interpolate.interp1d(xgrid, zgrid,
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=0.0)
    zgrid = interp_z(reg_xgrid)
    xgrid = reg_xgrid
    print(len(xgrid), len(ygrid), len(zgrid))
    np.savetxt(output, np.c_[xgrid, ygrid, zgrid], header="xaxis, " \
                                                          "combined y axis, " \
                                                          "filtered y axis")

def parse_args():
    """
    """
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('specs', type=str, nargs='+',
                        help="Spectra to process (string).")
    parser.add_argument('stack', type=str,
                        help="Output stack filename (string).")
    parser.add_argument('-m', '--mode', type=str, default='interpol',
                        help="Mode used to stack.\n" \
                             "Available: interpol or filter\n" \
                             "Default: interpol\n" \
                             "If mode is interpol, then weight and weight_list\n" \
                             "options are also used.\n" \
                             "If mode is filter, then window and window_opts\n" \
                             "options are used.")
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
    parser.add_argument('--weight_list_cols', type=str, default='0,1',
                        help="Columns to use from the file with the weights.\n" \
                             "Default: 0,1")
    parser.add_argument('--window', type=str, default='Gauss',
                        help="Window function to filter noise in stacked spectrum.\n" \
                             "Only used if mode is filter.\n" \
                             "Can be one of: Gauss, SavGol and Wiener.\n" \
                             "Default: Gauss")
    parser.add_argument('--window_opts', type=str, default='sigma=3,order=0',
                        help="Window function options.\n" \
                             "Only used if mode is filter.\n" \
                             "Default: 'sigma=3,order=0'")
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
    parser.add_argument('--x_col', type=int, default=0,
                        help="Column with x axis values. Default: 0")
    parser.add_argument('--y_col', type=int, default=1,
                        help="Column with y axis values. Default: 1")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose output?")
    parser.add_argument('-l', '--logfile', type=str, default=None,
                        help="Where to store the logs.\n" \
                             "(string, Default: output to console)")
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.verbose:
        loglev = logging.DEBUG
    else:
        loglev = logging.ERROR
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=args.logfile, level=loglev, format=formatter)
    
    logger = logging.getLogger(__name__)
    
    if args.mode == 'interpol':
        if args.weight == 'list' and not args.weight_list:
            logger.error("No weight list given.")
            logger.error("Will now exit.")
            sys.exit()
        else:
            stack_interpol(args.specs, args.stack, args.v_max, args.v_min, args.dv, 
                           args.x_col, args.y_col, 
                           args.weight, args.weight_list, args.weight_list_cols)
            
    elif args.mode == 'filter':
        waux = args.window_opts.split(',')
        window_opts = {}
        for aux in waux:
            auxk, auxv = aux.split('=')
            window_opts[auxk] = float(auxv)
        print(window_opts)
        stack_filter(args.spec, args.stack, args.v_max, args.v_min, args.dv,
                     args.x_col, args.y_col, args.window, window_opts)
    else:
        logger.error("Mode not recognized.")
        logger.error("Will exit now.")
        sys.exit()
