#!/usr/bin/env python

import matplotlib as mpl
mpl.use('pdf')
import pylab as plt
import numpy as np
import glob
import sys
import argparse
from crrlpy import crrls
from scipy import interpolate


def show_coverage(spec, out, x_col, y_col, t_col):
    """
    Shows the coverage of spec in percentage.
    """
    
    data = np.loadtxt(spec)
    x = data[:,x_col]
    c = data[:,y_col]
    t = data[:,t_col]
    
    if abs(t.max()) < abs(t.min()):
        tnorm = abs(t.min())
    else:
        tnorm = abs(t.max())
    
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(1, 1, 1, adjustable='datalim')
    ax.step(x, abs(c - c.max())/c.max()*100, 'k-',  
            drawstyle='steps', lw=1, where='pre', label='coverage')
    ax.step(x, t/tnorm*100, '-', c='gray', 
            drawstyle='steps', lw=1, where='pre', label='spectrum')
    ax.set_xlabel(r'Velocity (km s$^{-1}$)')
    ax.set_ylabel(r'Percentage $\%$')
    ax.legend(loc=0, numpoints=1, frameon=False)
    plt.savefig('{0}'.format(out), 
                bbox_inches='tight', pad_inches=0.3)
    plt.close()

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', type=str,
                        help="File with coverage to show.\n" \
                             "E.g., \"lba_hgh_*.ascii\" (string).\n" \
                             "Wildcards and [] accepted.")
    parser.add_argument('out', type=str,
                        help="Output plot filename.\n" \
                             "E.g., CIalpha_stack1_coverage.pdf (string).")
    parser.add_argument('--x_col', type=int, default=0,
                        help="Column with x axis values.\n" \
                             "Default: 0")
    parser.add_argument('--y_col', type=int, default=2,
                        help="Column with y axis values.\n" \
                             "Default: 2")
    parser.add_argument('--t_col', type=int, default=1,
                        help="Column with optical depth values.\n" \
                             "Default: 1")
    args = parser.parse_args()
    
    show_coverage(args.spec, args.out, args.x_col, args.y_col, args.t_col)