#!/usr/bin/env python

import sys
import argparse
import numpy as np
from astropy.table import Table
from astropy.io import ascii

def main(spec, out):
    """
    """
    
    data = np.loadtxt(spec, comments='#')
    
    freq = data[:,0]/1e6
    tb = data[:,1]
    
    # write the processed spectrum
    tbtable = Table([freq, tb], 
                    names=['FREQ MHz',
                           'Tb Jy/BEAM'])

    ascii.write(tbtable, out, format='commented_header')
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', 
                        help="Spectrum to process.")
    parser.add_argument('-o', '--outspec', 
                        help="Output spectrum name.", 
                        type=str, dest='out', required=True)
    args = parser.parse_args()
    
    spec = args.spec
    out = args.out
        
    main(spec, out)