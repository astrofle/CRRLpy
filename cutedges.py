#!/usr/bin/env python

import numpy as np
import glob
import re
import sys
import argparse

def cut_edges(spec, basename, mode, **kwargs):
    """
    Removes the edges from the spectra specified by 
    spec and saves them in new files with names given 
    by basename_SBXXX.ascii.
    """
    
    specs = glob.glob(spec)
    
    if mode == 'persb':
        elist = np.loadtxt(kwargs['edge_list'], dtype='|S5,<i5')
    
    for s in specs:
        
        # Determine the subband name
        sb = re.findall('SB\d+', s)[0]

        if mode == 'persb':
            try:
                edge = elist['f1'][np.where(elist['f0'] == sb)[0][0]]
            except IndexError:
                print "{0} missing from {1}.".format(sb, kwargs['edge_list'])
                print "Exiting now."
                sys.exit()
        else:
            edge = kwargs['edge']
        
        data = np.loadtxt(s)
        # Select the data ignoring the edges
        data = data[edge:-edge]

        np.savetxt("{0}_{1}.ascii".format(basename, sb), data)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', type=str,
                        help="Files with spectrum to process. " \
                             "(string, wildcards and [] accepted)")
    parser.add_argument('basename', type=str,
                        help="Base name of output files. " \
                             "e.g., <basename>_SB121.ascii")
    parser.add_argument('-m', '--mode', type=str, default='const',
                        help="Type of cut to apply.\n" \
                             "const: Use a constant number of channels.\n" \
                             "persb: Use a user defined list.\n"\
                             "Default: const")
    parser.add_argument('-l', '--edge_list', type=str, default=None,
                        help="List with edges to remove for each subband.\n" \
                             "It should be of the form:\n" \
                             "SB000 14\nSB001 13\nSB002 12\n  ...")
    parser.add_argument('-e', '--edge', type=int, default=1,
                        help="Channels to remove. Only used if mode=const. " \
                             "Default: 1")
    args = parser.parse_args()

    if args.mode == 'persb' and args.edge_list is None:
        print "A list with edges is required for this mode."
        sys.exit()
                        
    cut_edges(args.spec, args.basename, args.mode, 
              edge=args.edge, edge_list=args.edge_list)