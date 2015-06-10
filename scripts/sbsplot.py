#!/usr/bin/env python

import glob
import re
import argparse
import os

import matplotlib as mpl
havedisplay = "DISPLAY" in os.environ
if not havedisplay:
    mpl.use('Agg')
import pylab as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from crrlpy import crrls

#colors = ['r', 'orange', 'g', 'b']
tprops = {'CIalpha':[r'C$\alpha$', 'r'],
          'CIbeta':[r'C$\beta$', 'orange'],
          'CIgamma':[r'C$\gamma$', 'g'],
          'CIdelta':[r'C$\delta$', 'b'],
          'HIalpha':[r'H$\alpha$', 'y'],
          'NIalpha':[r'N$\alpha$', 'gray']
          }
ylbl = 1e-4

def sbsplot(spec, output, show_lines, transitions, z, x_axis, x_col, y_axis, y_col):
    """
    """
    
    pdf = PdfPages(output)
    
    specs = glob.glob(spec)
    crrls.natural_sort(specs)
    
    # If only one file is passed, it probably contains a list
    if len(specs) == 1:
        specs = np.genfromtxt(specs[0], dtype=str)
        try:
            specs.shape[1]
            specs = glob.glob(spec)
        # Or a single file is to be plotted
        except IndexError:
            pass

    for s in specs:
                
        data = np.loadtxt(s)
        x = data[:,x_col]
        y = data[:,y_col]
        
        # Determine the subband name
        try:
            sb = re.findall('SB\d+', s)[0]
        except IndexError:
            print "Could not find SB number."
            print "Will use SB???"
            sb = 'SB???'
        
        # Begin ploting      
        fig = plt.figure(frameon=False)
        fig.suptitle(sb)
        ax = fig.add_subplot(1, 1, 1, adjustable='datalim')
        ax.step(x, y, 'k-', drawstyle='steps', lw=1, where='pre')
        # Mark the transitions?
        if show_lines:
            trans = transitions.split(',')
            for o,t in enumerate(trans):
                qns, freqs = crrls.find_lines_sb(x[~np.isnan(x)], t, z)
                for label, i, j in zip(qns, freqs, [ylbl]*len(freqs)):
                    plt.annotate(label, xy=(i, j), xytext=(-10, 15*o+5), 
                                 size='x-small', textcoords='offset points', 
                                 ha='right', va='bottom', 
                                 bbox=dict(boxstyle='round,pad=0.5', 
                                         fc='yellow', alpha=0.5),
                                 arrowprops=dict(arrowstyle='->', 
                                                 connectionstyle='arc3,rad=0'))
                    #if len(qns) > 0:
                    plt.annotate(tprops[t][0], xy=(i,j), xytext=(-4,0), 
                                textcoords='offset points', size='xx-small')
                    plt.plot(freqs, [ylbl]*len(freqs), marker='|', ls='none', ms=25, 
                            c=tprops[t][1], mew=8, alpha=0.8)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        pdf.savefig(fig)
        plt.close(fig)
                
    pdf.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=str,
                        help="Files with spectrum to process." \
                             "(string, wildcards and [] accepted)")
    parser.add_argument('output', type=str,
                        help="Name of output file with spectrum plots." \
                             "Will produce a single .pdf file.")
    parser.add_argument('-l', '--show_lines', action='store_true',
                        help="Show lines in the spectra? Default: False")
    parser.add_argument('-t', '--transitions', type=str, default='CIalpha',
                        help="Transitions to show in the spectra." \
                             "E.g., CIalpha,CI13beta,HIalpha" \
                             "Default: CIalpha")
    parser.add_argument('--z', type=float, default=0.0, dest='z',
                        help="Redshift to apply to the transition rest frequency." \
                             "Default: 0")
    parser.add_argument('-x', '--x_axis', type=str, default='Frequency (MHz)',
                        help="X axis of the spectra." \
                             "Default: Frequency (MHz)")
    parser.add_argument('--x_col', type=int, default=0,
                        help="Column with x axis values. Default: 0")
    parser.add_argument('-y', '--y_axis', type=str, default='Optical depth',
                        help="Y axis of the spectra." \
                             "Default: Optical depth")
    parser.add_argument('--y_col', type=int, default=1,
                        help="Column with y axis values. Default: 1")
    args = parser.parse_args()
    
    sbsplot(args.spec, args.output, 
            args.show_lines, args.transitions, args.z,
            args.x_axis, args.x_col,
            args.y_axis, args.y_col)