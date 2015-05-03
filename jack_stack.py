#!/usr/bin/env python

"""
Reads in a list with N spectra and makes new lists leaving out a number
M of spectra from the initial list.
M < N
This for testing if a single line dominates a stack, or to evaluate how
the line parameters changes with the stacked lines.
"""

import numpy as np
import argparse
import sys

def jack_list(spec_list, M, basename):
    """
    Reads a list of lenght N and leaves 1 element out.
    spec_list should be a text file with a list of N files.
    M<N
    The list with the selected elements will be <basename>_jx.log,
    with x the subsample number.
    """
    
    specs = np.genfromtxt(spec_list, dtype=str)
    
    if M >= len(specs):
        print "Left out elements larger than the sample."
        print "No possible subsample."
        print "Will now exit."
        sys.exit()
        
    for i in xrange(len(specs)):
        sspecs = np.array([s for j,s in enumerate(specs) if j != i])
        np.savetxt("{0}_j{1:.0f}.log".format(basename, i), sspecs, fmt="%s")
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec_list', type=str,
                        help="List of spectra to process.\n" \
                             "Should be a file with the list.\n" \
                             "E.g., \"lba_hgh_*.ascii\" (string).")
    parser.add_argument('basename', type=str,
                        help="Base name of output files.\n" \
                             "E.g., <basename>_jx.log with x the subsample number.")
    parser.add_argument('-m', type=int, default=1,
                        help="Number of elements to leave out in each subsample.\n" \
                             "M<N\n" \
                             "Not yet implemented.\n"\
                             "Default: 1")
    args = parser.parse_args()
    
    jack_list(args.spec_list, args.m, args.basename)