#!/usr/bin/env python

from lmfit import Model
import numpy as np
import sys
import argparse

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

def remove_baseline(freq, tb, model, p0, mask):
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

    fit = mod.fit(tb[~mask], x=freq[~mask], params=params)
    #best_fit = mod.eval(x=freq, amp=10, cen=6.2, wid=0.75)
    tbcsub = tb/fit.eval(x=freq) - 1
    
    return tbcsub

def main(spec_in, spec_out, edge):
    """ main """
    
    # Load the data
    data = np.loadtxt(spec_in)
    freq = data[:,0]
    tb = data[:,1]
    
    # Get a mask for NaN values
    mask = nan_mask(freq, tb)
    
    # Set the initial parameters to fit the continuum
    a = (tb[~mask][-1-edge] - tb[~mask][0+edge])/\
        (freq[~mask][-1-edge] - freq[~mask][0+edge])
    b = tb[~mask][0+edge]
    p0 = {'a':a, 'b':b}
    
    # Remove the continuum
    tbcsub = remove_baseline(freq, tb, linear, p0, mask)
    
    # Save the continuum subtracted spectrum
    np.savetxt(spec_out, np.c_[freq, tbcsub])
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', help="Spectrum to remove continuum from.")
    parser.add_argument('out', help="Output spectrum without continuum.")
    parser.add_argument('-e', '--edge', help="How many edge channels \
                        should be ignored?", dest='edge', type=int, default=0)
    args = parser.parse_args()
    
    spec = args.spec
    out = args.out
    edge = args.edge
    
    main(spec, out, edge)