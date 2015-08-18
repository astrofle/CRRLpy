#!/usr/bin/env python
import pylab as plt
import numpy as np
import glob
import re
import sys
import argparse
from scipy.interpolate import UnivariateSpline

def main(spec, basename, order, median, x_col, y_col, save, baseline):
    """
    """
    
    specs = glob.glob(spec)
    
    # If only one file is passed, it probably contains the list
    if len(specs) == 1:
        specs = np.genfromtxt(specs[0], dtype=str)
    
    for s in specs:
        
        # Determine the subband name
        try:
            sb = re.findall('SB\d+', s)[0]
        except IndexError:
            print "Could not find SB number."
            print "Will use SB???"
            sb = 'SB???'
        
        # Load the data
        data = np.loadtxt(s)
        x = data[:,x_col]
        y = data[:,y_col]        
        
        # Turn NaNs to zeros
        my = np.ma.masked_invalid(y)
        mx = np.ma.masked_where(np.ma.getmask(my), x)
        np.ma.set_fill_value(my, 0)
        np.ma.set_fill_value(mx, 0)
        gx = mx.compressed()
        gy = my.compressed()
        
        # Use a polynomial to remove the baseline
        bp = np.polynomial.polynomial.polyfit(gx, gy, order)
        b = np.polynomial.polynomial.polyval(gx, bp)
        #b = UnivariateSpline(gx, gy, k=order)
        
        if median:
            # Only keep the baseline shape
            #gb = b(x) - np.median(b(x))
            gb = b - np.median(b)
        else:
            #gb = b(x)
            gb = b

        if save:
            np.savetxt('{0}_{1}.ascii'.format(baseline, sb), 
                       np.c_[x, gb])#np.ma.masked_where(np.ma.getmask(mx), gb).compressed()])
        
        data[:,y_col] = y - gb
                
        np.savetxt('{0}_{1}.ascii'.format(basename, sb), data)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', type=str,
                        help="List of spectra to process.\n" \
                             "Can be a file with the list or a list.\n" \
                             "E.g., \"lba_hgh_*.ascii\" (string).\n" \
                             "Wildcards and [] accepted.")
    parser.add_argument('basename', type=str,
                        help="Basename of the baseline corrected spectra. (string)\n" \
                             "Will be of the form: <basename>_SB121.ascii\n" \
                             "The output will have the y column baseline corrected.")
    parser.add_argument('-k', '--order', type=int,
                        help="Spline order. (int<=5)")
    parser.add_argument('-m', '--median', action='store_true',
                        help="Remove the median from the baseline?")
    parser.add_argument('-x', '--x_col', type=int, default=0,
                        help="Column with x axis values. Default: 0")
    parser.add_argument('-y', '--y_col', type=int, default=1,
                        help="Column with y axis values. Default: 1")
    parser.add_argument('-s', '--save', action='store_true',
                        help="Save the baseline model?. Default: False")
    parser.add_argument('-b', '--baseline', type=str, default=None,
                        help="Basename for the baseline models. (string)\n" \
                             "Will be of the form: <basename>_SB121.ascii\n" \
                             "Default: None")
    args = parser.parse_args()
    
    if args.save and not args.baseline:
        print "No basename to save the baseline models."
        print "Will now exit."
        sys.exit()
    
    main(args.spec, args.basename, args.order, args.median, 
         args.x_col, args.y_col, args.save, args.baseline)