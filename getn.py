#!/usr/bin/env python

import numpy as np
import glob
import re
import argparse

def get_n(line_list):
    """
    """
    
    ll = np.loadtxt(line_list, dtype=str)
    nl = np.zeros(len(ll))
    
    for i,l in enumerate(ll):
        l.split('/')[-1]
        nl[i] = int(re.findall('n\d+', l)[0][1:])
    
    print np.mean(nl)
    if nl[0] < nl[-1]:
        print "{0}-{1}".format(nl[0], nl[-1])
    else:
        print "{0:.0f}-{1:.0f}".format(nl[-1], nl[0])
    print len(nl)
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('line_list', type=str,
                        help="Name of the file with good lines. (string)")
    args = parser.parse_args()
    
    get_n(args.line_list)