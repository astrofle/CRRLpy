#!/usr/bin/env python

import sys
import argparse
import os
import matplotlib as mpl
havedisplay = "DISPLAY" in os.environ
if not havedisplay:
    mpl.use('Agg')

from lmfit import Model
from crrlpy import crrls

import pylab as plt
import numpy as np

def linear(x, a, b):
    """
    Linear model.
    """
    return a*x + b

def nan_mask(freq, tb):
    """
    Find a mask for NaN values
    """
    fnan = np.isnan(freq)
    tnan = np.isnan(tb)
    mask = reduce(np.logical_or, [fnan, tnan])
    
    return mask

def fit_baseline(freq, tb, model, p0):
    """
    """
    mod = Model(model)
    params = mod.make_params()
    if len(p0) != len(params):
        print "Insuficient starting parameter values."
        return 0
    else:
        for param in params:
            params[param].set(value=p0[param])

    return mod.fit(tb, x=freq, params=params)
    

def remove_baseline(freq, tb, model, p0):
    """
    Divide tb by given a model
    and starting parameters p0.
    Returns: tb/model - 1
    """
    # Divide by linear baseline
    mod = Model(model)
    params = mod.make_params()
    if len(p0) != len(params):
        print "Insuficient starting parameter values."
        return 0
    else:
        for param in params:
            params[param].set(value=p0[param])

    fit = mod.fit(tb, x=freq, params=params)
    tbcsub = tb/fit.eval(x=freq) - 1
    
    return tbcsub

def main(spec_in, spec_out, edge, xcol, ycol, transitions, z, dv, plot, plot_out):
    """ main """
    
    # Load the data
    data = np.loadtxt(spec_in)
    freq = data[:,xcol]
    tb = data[:,ycol]
    
    # Get a mask for NaN values
    mask = nan_mask(freq, tb)
    
    # Remove NaNs and invalid values
    x = freq[~mask]
    y = tb[~mask]
    
    # Blank lines
    trans = transitions.split(',')
    bf = []
    for o,t in enumerate(trans):
        n, f = crrls.find_lines_sb(x, t, z)
        bf.append(list(f))
    if len(bf) > 0:
        bf = np.array(list(_f for _bf in bf for _f in _bf))
        x_lf, y_lf = crrls.blank_lines2(x, y, bf, dv)
    else:
        x_lf, y_lf = x,y
    
    # Set the initial parameters to fit the continuum
    a = (tb[~mask][-1-edge] - tb[~mask][0+edge])/\
        (freq[~mask][-1-edge] - freq[~mask][0+edge])
    b = tb[~mask][0+edge]
    p0 = {'a':a, 'b':b}
    
    # Remove the continuum
    #tbcsub = remove_baseline(freq[~mask], tb[~mask], linear, p0)
    baseline = fit_baseline(x_lf, y_lf, linear, p0)
    tbcsub = tb/baseline.eval(x=freq) - 1
    
    if plot:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(freq, tb, 'k-', label='in')
        ax.plot(freq, baseline.eval(x=freq), 'r-', label='continuum')
        ax.plot(x_lf, y_lf, 'b-', label='line free spectra')
        plt.legend(loc=0, numpoints=1, frameon=False)
        plt.savefig('{0}'.format(plot_out), 
                bbox_inches='tight', pad_inches=0.3)
        plt.close()
    
    # Save the continuum subtracted spectrum
    np.savetxt(spec_out, np.c_[freq, tbcsub])
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', help="Spectrum to remove continuum from.")
    parser.add_argument('out', help="Output spectrum without continuum.")
    parser.add_argument('-e', '--edge', help="How many edge channels \
                        should be ignored?", dest='edge', type=int, default=0)
    parser.add_argument('--x_col', type=int, default=0,
                        help="Column with x axis values. Default: 0")
    parser.add_argument('--y_col', type=int, default=1,
                        help="Column with y axis values. Default: 1")
    parser.add_argument('-t', '--transitions', type=str, default='CIalpha',
                        help="Transitions to blank in the spectra. (string)\n" \
                             "E.g., CIalpha,CI13beta,HIalpha\n" \
                             "Default: CIalpha.")
    parser.add_argument('--z', type=float, default=0.0, dest='z',
                        help="Redshift to apply to the transition rest frequency. (float)\n" \
                             "Default: 0")
    parser.add_argument('-d', '--dv', type=float, default=50.,
                        help="Velocity range to blank around each transition \n" \
                             "in km/s. (float, km/s)\n" \
                             "Default: 50")
    parser.add_argument('--plot', action='store_true',
                        help="Plot the generated spectrum?")
    parser.add_argument('--plot_out', type=str,
                        help="Plot name. (string)")
    args = parser.parse_args()
    
    spec = args.spec
    out = args.out
    edge = args.edge
    xcol = args.x_col
    ycol = args.y_col
    transitions = args.transitions
    z = args.z
    dv = args.dv
    plot = args.plot
    plot_out = args.plot_out
    
    main(spec, out, edge, xcol, ycol, transitions, z, dv, plot, plot_out)