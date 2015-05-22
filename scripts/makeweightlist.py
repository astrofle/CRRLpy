#!/usr/bin/env python

import numpy as np
import glob
import argparse
from crrlpy import crrls

def set_weight(mode, rms):
    """
    Sets the type of weight to output.
    Change this to add more weight definitions.
    """
    
    if mode == 'rms':
                mrms = rms
    if mode == 'rms2':
        mrms = np.power(rms, 2)
    if mode == '1/rms':
        mrms = 1./rms
    if mode == '1/rms2':
        mrms = 1./np.power(rms, 2)
        
    return mrms

def make_rms_list(spec, output, transitions, z, dv, mode, f_col, y_col):
    """
    """
    
    specs = glob.glob(spec)
    crrls.natural_sort(specs)
    
    with open(output, 'w') as log:
        for s in specs:
                        
            data = np.loadtxt(s)
            x = data[:,f_col]
            y = data[:,y_col]
            
            # Catch NaNs and invalid values:
            mask_x = np.ma.masked_equal(x, 1.0).mask
            mask_y = np.isnan(y)
            mask = np.array(reduce(np.logical_or, [mask_x, mask_y]))
            
            # Remove NaNs and invalid values
            x = x[~mask]
            y = y[~mask]
            
            trans = transitions.split(',')
            bf = []
            for o,t in enumerate(trans):
                n, f = crrls.find_lines_sb(x, t, z)
                bf.append(list(f))
            if len(bf) > 0:
                bf = np.array(list(_f for _bf in bf for _f in _bf))
                x_lf, y_lf = crrls.blank_lines2(x, y, bf, dv)
            else:
                x_lf, y_lf = x,y
            
            rms = crrls.get_rms(y_lf)
            
            mrms = set_weight(mode, rms)
            
            # Get the SB frequency
            freq = np.mean(x)
                
            log.write("{0}  {1}   {2}\n".format(s, mrms, freq))  
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', type=str,
                        help="List of spectra to process.\n" \
                             "Can be a file with the list or a list.\n" \
                             "E.g., \"lba_hgh_*.ascii\" (string).\n" \
                             "Wildcards and [] accepted.")
    parser.add_argument('output', type=str,
                        help="Name of the output file. (string)")
    parser.add_argument('-m', '--mode', type=str, default='1/sigma',
                        help="Type of output. (string)\n" \
                             "Can be: rms, rms2, 1/rms, 1/rms2.\n" \
                             "Default: 1/rms")
    parser.add_argument('-t', '--transitions', type=str, default='CIalpha',
                        help="Transitions to blank in the spectra. (string)\n" \
                             "E.g., CIalpha,CI13beta,HIalpha\n" \
                             "Default: CIalpha.")
    parser.add_argument('--z', type=float, default=0.0, dest='z',
                        help="Redshift to apply to the transition rest frequency. (float)\n" \
                             "Default: 0")
    parser.add_argument('-d', '--dv', type=float, default=50.,
                        help="Velocity range to blank around each transition \n" \
                             "in km/s. (float, km/s)\n" \
                             "Default: 50")
    parser.add_argument('--f_col', type=int, default=0,
                        help="Column with frequency values. Default: 0")
    parser.add_argument('--y_col', type=int, default=1,
                        help="Column with y axis values. Default: 1")
    args = parser.parse_args()
    
    make_rms_list(args.spec, args.output, args.transitions, args.z, args.dv, 
                  args.mode, args.f_col, args.y_col)