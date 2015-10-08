#!/usr/bin/env python

import matplotlib as mpl
mpl.use('pdf')
import pylab as plt
import numpy as np
import glob
import re
import sys
import argparse
from crrlpy import crrls
from scipy import interpolate
from matplotlib.backends.backend_pdf import PdfPages

def remove_stack(spec, model, basename, transition, z, x_col, y_col, freq, 
                 plot, plot_file):
    """
    """
    
    specs = glob.glob(spec)
    crrls.natural_sort(specs)
    
    # If only one file is passed, it probably contains a list
    if len(specs) == 1:
        specs = np.genfromtxt(specs[0], dtype=str)
        try:
            specs.shape[1]
            specs = glob.glob(spec)
        # Or a single file is to be processed
        except IndexError:
            pass
        
    if plot:
        pdf = PdfPages(plot_file)
    
    for s in specs:
        
        # Determine the subband name
        try:
            sb = re.findall('SB\d+', s)[0]
        except IndexError:
            print "Could not find SB number."
            print "Will use SB???"
            sb = 'SB???'
        
        data = np.loadtxt(s)
        p = data[:,x_col].argsort()
        x = np.copy(data[p,x_col])
        y = np.copy(data[p,y_col])
        
        mask = np.isnan(y)
        
        # Load model
        mod = np.loadtxt(model)
        p = mod[:,0].argsort()
        xm = mod[p,0]
        ym = mod[p,1]
        # remove NaNs
        ym = ym[~np.isnan(xm)]
        xm = xm[~np.isnan(xm)]
        
        qns, freqs = crrls.find_lines_sb(x, transition, z)
        #print qns
        
        y_mod = np.zeros(len(y))
        ys = np.copy(y)
        ys[mask] = 0
        x[mask] = -9999 # This way it should be outside the boundaries
        if not freq:
            for i,n in enumerate(qns):
                # Convert the model velocity axis to frequency
                fm = crrls.vel2freq(freqs[i]*(1.+z), xm*1e3)
                p = fm.argsort()
                ymod = ym[p]
                fm = fm[p]
                
                # Interpolate the model axis to the spectrum grid
                interp_ym = interpolate.interp1d(fm, ymod,
                                                kind='linear',
                                                bounds_error=False,
                                                fill_value=0.0)
                y_mod += interp_ym(x)
                                
        else:
            # Interpolate the model axis to the spectrum grid
            interp_ym = interpolate.interp1d(xm, ym,
                                            kind='linear',
                                            bounds_error=False,
                                            fill_value=0.0)
            y_mod += interp_ym(x)
        
        # Remove the model
        ys = ys - y_mod
        
        # Return the masked values to their NaN values
        ys[mask] = np.nan
        x[mask] = np.nan
        
        if plot:
            fig = plt.figure(frameon=False)
            fig.suptitle(sb)
            ax = fig.add_subplot(1, 1, 1, adjustable='datalim')
            ax.step(x, y_mod, 'r-', drawstyle='steps', lw=1, where='pre', label='model')
            ax.step(x, y, 'b-', drawstyle='steps', lw=1, where='pre', label='in')
            ax.step(x, ys, 'g-', drawstyle='steps', lw=1, where='pre', label='out')
            ax.legend(loc=0, numpoints=1, frameon=False)
            pdf.savefig(fig)
            plt.close(fig)

        data[:,y_col] = ys
        np.savetxt('{0}_{1}.ascii'.format(basename, sb), data)
        
    if plot:
        pdf.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', type=str,
                        help="List of spectra to process.\n" \
                             "Can be a file with the list or a list.\n" \
                             "E.g., \"lba_hgh_*.ascii\" (string).\n" \
                             "Wildcards and [] accepted.")
    parser.add_argument('model', type=str,
                        help="Filename of model to remove (string).\n"\
                             "The model should have a velocity (km/s) \n" \
                             "and intesity axis. E.g.,\n" \
                             "# Velocity (km/s) Tau\n" \
                             "190.0 0.0001\n" \
                             "189.0 0.00008\n" \
                             "...   ...")
    parser.add_argument('basename', type=str,
                        help="Output spectra basename (string).\n" \
                             "The resulting spectra will be named: \n" \
                             "<basename>_SBxxx.ascii")
    parser.add_argument('-t', '--transition', type=str, default='CIalpha',
                        help="Transition being removed. (string)\n" \
                             "E.g., CI13beta\n" \
                             "Default: CIalpha.")
    parser.add_argument('--z', type=float, default=0.0, dest='z',
                        help="Redshift to apply to the transition rest frequency." \
                             "Default: 0")
    parser.add_argument('--x_col', type=int, default=0,
                        help="Column with x axis values in spectra. Default: 0")
    parser.add_argument('--y_col', type=int, default=1,
                        help="Column with y axis values in spectra. Default: 1")
    parser.add_argument('-f', '--freq', action='store_true',
                        help="Is the x axis in frequency? Default: False")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="Plot the spectra? Default: False")
    parser.add_argument('--plot_file', type=str, default=None,
                        help="Plot file name. Default: None")
    parser.add_argument('-i', '--interp', type=str, default='linear',
                        help="Kind of interpolation to use with scipy.interpolate.interp1d." \
                             "Default: linear")
    args = parser.parse_args()
    
    if args.plot and not args.plot_file:
        print "I do not know where to save the plot."
        print "Will now exit."
        sys.exit()
    
    remove_stack(args.spec, args.model, args.basename, args.transition, args.z,
                 args.x_col, args.y_col, args.freq, args.plot, args.plot_file)