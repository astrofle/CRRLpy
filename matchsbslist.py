#!/usr/bin/env python

"""
Lazy, lazy astronomers!
"""

import numpy as np
import re
import sys
import argparse

def complete_list(first_sb, last_sb, sb_list, output):
    """
    
    """
    # Just to make the code readable and avoid
    # duplicating it.
    
    i0 = len(sb_list) + 1
    with open(output, 'w') as log:
        for i,s in enumerate(sb_list):
            if first_sb in s:
                i0 = i
            if i >= i0:
                log.write("{0}\n".format(s))
            if last_sb in s:
                break

def main(line_list, sbs_list, output):
    """
    """
    
    ll = np.loadtxt(line_list, dtype=str)
    sl = np.loadtxt(sbs_list, dtype=str)
    
    first_sb = re.findall('SB\d+', ll[0])[0]
    last_sb = re.findall('SB\d+', ll[-1])[0]
    
    try:
        complete_list(first_sb, last_sb, sl[:,0], output)
    except IndexError:
        complete_list(first_sb, last_sb, sl, output)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('line_list', type=str,
                        help="Name of the file with lines. (string)")
    parser.add_argument('sbs_list', type=str,
                        help="Name of the file with all the available SBs. (string)")
    parser.add_argument('output', type=str,
                        help="Name of the output file. (string)")
    args = parser.parse_args()
    
    main(args.line_list, args.sbs_list, args.output)
    