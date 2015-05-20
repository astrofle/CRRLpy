#!/usr/bin/env python

import numpy as np
import argparse
from crrlpy import crrls

def remove_outliers(spec, output, thr, col):
    """
    """
    
    data = np.loadtxt(spec)
    y = data[:,col]
    
    # Remove outliers
    mask = crrls.mask_outliers(y, m=thr)
    #tb = tb[~mask]
    #freq = freq[~mask]
    
    data = data[~mask]
    
    np.savetxt(output, data)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', type=str,
                        help="Spectrum to process. (string)")
    parser.add_argument('output', type=str,
                        help="Name of the output file. (string)")
    parser.add_argument('-t', '--threshold', type=float,
                        help="Threshold to clip outliers. (float)")
    parser.add_argument('-c', '--col', type=int, default=1,
                        help="Column with values to clip. Default: 1")
    args = parser.parse_args()
    
    remove_outliers(args.spec, args.output, args.threshold, args.col)