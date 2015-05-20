#!/usr/bin/env python

"""
Given a weight per subband, created with rmslist.py,
this script will match the corresponding weight of
a given subband to the lines in the given list.
Both lists should have a subband number preceded by
SBxxx.
"""

import numpy as np
import re
import sys
import argparse

def find_substring(array, substr):
    """
    """
    
    for i,s in enumerate(array):
        if substr in s:
              return i
    return -1

def match_weights(weights, lines, output, dup=False):
    """
    """
    
    wl = np.loadtxt(weights, dtype='|S99,<f10')
    ll = np.loadtxt(lines, dtype='|S99')
    
    with open(output, 'w') as log:
        for i,l in enumerate(ll):
            
            # Determine the subband name
            try:
                sb = re.findall('SB\d+', l)[0]
            except IndexError:
                print "Could not find SB number."
                print "This will not work."
                print "Exiting now."
                sys.exit()
            if not dup:
                j = find_substring(wl['f0'], sb)
            else:
                j = find_substring(wl['f0'], l.split('_')[1]+'_'+sb)
            log.write("{0} {1}\n".format(l, wl['f1'][j]))
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('weights', type=str,
                        help="Name of the file with SB weights. (string)")
    parser.add_argument('lines', type=str,
                        help="Name of the file with lines. (string)")
    parser.add_argument('output', type=str,
                        help="Name of the output file. (string)")
    parser.add_argument('-d', '--duplicate', action='store_true',
                        help="Are there duplicated subband numbers?. (string)")
    args = parser.parse_args()
    
    match_weights(args.weights, args.lines, args.output, args.duplicate)