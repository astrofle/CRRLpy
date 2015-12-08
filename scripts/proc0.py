#!/usr/bin/env python

import sys
import argparse
import glob
import re
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from crrlpy import crrls

def main(spec, out, freqf):
    """
    """
    
    specs = glob.glob(spec)
    
    # Sort the SBs
    crrls.natural_sort(specs)
    
    for s in specs:
        
        ## Determine the subband name
        #try:
            #sb = re.findall('SB\d+', s)[0]
        #except IndexError:
            #print "Could not find SB number."
            #print "Will use SB???"
            #sb = 'SB???'
        
        data = np.loadtxt(s, comments='#')
        

        freq = data[:,0]*freqf
        tb = data[:,1]

        data[:,0] = data[:,0]/1e6

        
        # write the processed spectrum
        #np.savetxt('{0}_{1}.ascii'.format(basename, sb), data)
        tbtable = Table([freq, tb], 
                        names=['FREQ MHz',
                               'Tb Jy/BEAM'])
                        
    ascii.write(tbtable, out, format='commented_header')
    
if __name__ == '__main__':
    
	parser = argparse.ArgumentParser()
	parser.add_argument('spec', 
						help="Spectrum to process.")
	parser.add_argument('-o', '--outspec', 
						help="Output spectrum base name.", 
						type=str, dest='out', required=True)
	parser.add_argument('-f', '--freqf', type=float, default=1e-6,
						help='Factor to multiply frequency with.')
	args = parser.parse_args()
	
	spec = args.spec
	out = args.out
	freqf = args.freqf
		
	main(spec, out, freqf)
	
