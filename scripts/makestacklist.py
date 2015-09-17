#!/usr/bin/env python

"""
If the SB list (sbs_list) has only one entry, then the script will fail.
The easiest way to avoid this is to add a dummy line to the file.
"""

import numpy as np
import glob
import re
import argparse

def fill_sb_list(first_sb, last_sb, sb_list, output, path):
    """
    """
    
    i0 = len(sb_list) + 1
    with open(output, 'w') as log:
        for i,s in enumerate(sb_list):
            if first_sb in s:
                i0 = i
            if last_sb in s:
                break
            if i >= i0:
                if not path:
                    log.write("{0}\n".format(s))
                else:
                    log.write("{0}/{1}\n".format(path, s.split('/')[-1]))
            

def main(good_list, sbs_list, basename, n, priority, path, first):
    """
    """
    
    sl = np.loadtxt(sbs_list, dtype=str)
    gl = np.loadtxt(good_list, dtype=str)
    nl = len(gl)
    ns = map(int, np.ones(n)*int(round(nl/n)))
    i = 0
    while int(nl - sum(ns)) > 0:
        if priority == 'First':
            ns[i] += 1
        elif priority == 'Last':
            ns[-i] += 1
        i += 1
    
    # Save the start point of every stack
    fn = np.array([first]*n)
    # Create the list of lines for every stack
    for i in xrange(n):
        fn[i] = re.findall('SB\d+', gl[sum(ns[:i])])[0]
        with open("{0}_stack{1}.log".format(basename, i+1), 'w') as log:
            for j in xrange(sum(ns[:i]),ns[i]+sum(ns[:i])):
                log.write("{0}\n".format(gl[j]))

    first_sb = first
    for i in xrange(n):
        output = "{0}_stack{1}_SBs.log".format(basename, i+1)        
        if i == n - 1:
            #print re.findall('SB\d+', sl[-1])
            try:
                sb = re.findall('SB\d+', sl[-1])[0]
            except TypeError:
                sb = re.findall('SB\d+', sl[-1,0])[0]
            last_sb = 'SB' + str(int(sb[2:])+1)
        else:
            last_sb = fn[i+1]
        try:
            fill_sb_list(first_sb, last_sb, sl[:,0], output, path)
        except IndexError:
            fill_sb_list(first_sb, last_sb, sl, output, path)
        if i != n - 1:
            first_sb = fn[i+1]
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('good_list', type=str,
                        help="Name of the file with good lines. (string)")
    parser.add_argument('sbs_list', type=str,
                        help="Name of the file with all the available SBs. (string)")
    parser.add_argument('basename', type=str,
                        help="Name of the output file. (string)")
    parser.add_argument('nstacks', type=int,
                        help="Number of stacks to produce. (int)")
    parser.add_argument('-p', '--priority', type=str, default='First',
                        help="Which stack should have more priority?\n" \
                             "First or Last. (string)")
    parser.add_argument('--path', type=str, default=None,
                        help="Path to use for the list of SBs. (string)\n" \
                             "Must be the path to the SBs that will have\n" \
                             "the transition removed.")
    parser.add_argument('--first', type=str, default='SB000',
                        help="First sub band in the data. (string)\n" \
                             "Useful if the sub band numbers do not start at 000.")
    args = parser.parse_args()
    
    main(args.good_list, args.sbs_list, args.basename, args.nstacks, 
         args.priority, args.path, args.first)