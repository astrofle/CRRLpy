#!/usr/bin/env python

import numpy as np
import glob
import re
import argparse
from crrlpy import crrls

def find_good_lines(spec, basename, transition, transs, z, vel_shift, x_col):
    """
    """
    
    specs = glob.glob(spec)
    crrls.natural_sort(specs)
    
    with open('{0}_good_lines.log'.format(transition), 'w') as log:
        for s in specs:
            
            # Determine the subband name
            sb = re.findall('SB\d+', s)[0]
            data = np.loadtxt(s)
            x = data[:,x_col]
            
            qns, freqs = crrls.find_lines_sb(x, transition, z)
            
            of = []
            for t in transs:
                n, f = crrls.find_lines_sb(x, t, z)
                of.append(list(f))
            of = np.array(list(_f for of_ in of for _f in of_))

            for i,freq in enumerate(freqs):
                vel = crrls.freq2vel(freq*1e6, freq*1e6)/1e3
                ovel = crrls.freq2vel(freq*1e6, of*1e6)/1e3
                diff = [abs(v - vel) for v in ovel]
                if all(d > vel_shift for d in diff):
                    log.write('{0}_{1}_n{2:.0f}.ascii\n'.format(basename, sb, qns[i]))
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=str,
                        help="Files with spectrum to process." \
                             "E.g., \"lba_hgh_*.ascii\" (string)." \
                             "Wildcards and [] accepted.")
    parser.add_argument('basename', type=str,
                        help="Base name for the output list. \n" \
                             "e.g., <basename>_SB120_n545.ascii")
    parser.add_argument('-t', '--transitions', type=str, default='RRL_CIalpha',
                        help="Transitions to consider in the spectra." \
                             "E.g., CIalpha,CI13beta,HIalpha" \
                             "Default: RRL_CIalpha")
    parser.add_argument('--z', type=float, default=0.0, dest='z',
                        help="Redshift to apply to the transition rest frequency." \
                             "Default: 0")
    parser.add_argument('-s', '--vel_shift', type=float, default=0.0,
                        help="Minimum velocity shift between lines in km/s." \
                             "Default: 0 km/s")
    parser.add_argument('--x_col', type=int, default=0,
                        help="Column with x axis values. Default: 0")
    args = parser.parse_args()
    
    trans = args.transitions.split(',')
    if len(trans) == 1:
        print "Only one transition being considered."
        print "Probably all lines will be good."
    
    for i,t in enumerate(trans):
        find_good_lines(args.spec, args.basename, t, trans[:i] + trans[i+1:],
                        args.z, args.vel_shift, args.x_col)