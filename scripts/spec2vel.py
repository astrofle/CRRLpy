#!/usr/bin/env python

"""
This script will convert a frequency axis to velocity
given a transition in the spectrum. The velocity axis will
be appended next to the frequency axis.
If the input has columns:     Frequency          Temperature RMS
The output will have columns: Frequency Velocity Temperature RMS
Columns before the frequency column will be lost.
"""

import numpy as np
import glob
import re
import argparse
from crrlpy import crrls

def spec2vel(spec, basename, transition, z, f_col, sb_id):
    """
    """
    
    specs = glob.glob(spec)
    
    # Sort the SBs
    crrls.natural_sort(specs)
    
    for s in specs:
        
        # Determine the subband name
        try:
            sb = re.findall('{0}\d+'.format(sb_id), s)[0]
        except IndexError:
            print "Could not find SB number."
            print "Will use SB???"
            sb = 'SB???'
        
        # Load the data
        data = np.loadtxt(s)
        x = data[:,f_col]
        y = data[:,f_col+1:]
        
        qns, freqs = crrls.find_lines_sb(x, transition, z)
        
        for i,n in enumerate(qns):
            # Convert the frequency axis to velocity
            vel = crrls.freq2vel(freqs[i]*(1+z), x)/1e3
            # Save the spectrum with a velocity column
            np.savetxt('{0}_{1}_n{2}.ascii'.format(basename, sb, int(n)), np.c_[x, vel, y])

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', type=str,
                        help="Files with spectrum to process.\n" \
                             "E.g., \"lba_hgh_*.ascii\" (string).\n" \
                             "Wildcards and [] accepted.")
    parser.add_argument('basename', type=str,
                        help="Base name of output files. \n" \
                             "e.g., <basename>_SB120_n545.ascii")
    parser.add_argument('-t', '--transition', type=str, default='CIalpha',
                        help="Transition to convert in the spectra.\n" \
                             "E.g., CI13beta" \
                             "Default: CIalpha")
    parser.add_argument('--z', type=float, default=0.0, dest='z',
                        help="Redshift to apply to the transition rest frequency.\n" \
                             "Default: 0")
    parser.add_argument('--f_col', type=int, default=0,
                        help="Column with frequency values.\n" \
                             "This will be converted to velocity. Default: 0")
    parser.add_argument('--sb_id', type=str, default='SB',
                        help="Column with frequency values.\n" \
                             "This will be converted to velocity. Default: 0")
    args = parser.parse_args()
    
    spec2vel(args.spec, args.basename, args.transition, args.z, args.f_col, args.sb_id)