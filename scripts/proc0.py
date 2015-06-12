#!/usr/bin/env python

import sys
import argparse
import glob
import re
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from crrlpy import crrls

def main(spec, basename):
    """
    """
    
    specs = glob.glob(spec)
    
    # Sort the SBs
    crrls.natural_sort(specs)
    
    for s in specs:
        
        # Determine the subband name
        try:
            sb = re.findall('SB\d+', s)[0]
        except IndexError:
            print "Could not find SB number."
            print "Will use SB???"
            sb = 'SB???'
        
        data = np.loadtxt(s, comments='#')
        
        freq = data[:,0]/1e6
        tb = data[:,1]
        
        # write the processed spectrum
        tbtable = Table([freq, tb], 
                        names=['FREQ MHz',
                            'Tb Jy/BEAM'])
                        
        out = '{0}_{1}.ascii'.format(basename, sb)

        ascii.write(tbtable, out, format='commented_header')
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', 
                        help="Spectrum to process.")
    parser.add_argument('-o', '--outspec', 
                        help="Output spectrum base name.", 
                        type=str, dest='out', required=True)
    args = parser.parse_args()
    
    spec = args.spec
    out = args.out
        
    main(spec, out)