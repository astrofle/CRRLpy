#!/usr/bin/env python

import numpy as np
import argparse
import glob

def pop_col(spec, col, basename=False):
    """
    Removes a column from the data in spec_list.
    """
    
    specs = glob.glob(spec)
    
    for spec in specs:
 
        data = np.loadtxt(spec)
        data = np.concatenate((data[:,:col], data[:,col+1:]), axis=1) 

        if basename:
            np.savetxt('{0}.log'.format(basename), data)
        else:
            np.savetxt(spec, data)
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', type=str,
                        help="List of spectra to process.\n" \
                             "Should be a file with the list.\n" \
                             "E.g., \"lba_hgh_*.ascii\" (string).")
    parser.add_argument('-b', '--basename', type=str, default=None,
                        help="Base name of output files.\n" \
                             "E.g., <basename>_jx.log with x the subsample number.")
    parser.add_argument('-c', '--column', type=int, default=1,
                        help="Column to remove from data.\n" \
                             "Default: 1")
    args = parser.parse_args()
    
    pop_col(args.spec, args.column, args.basename)
        
