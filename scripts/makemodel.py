#!/usr/bin/env python

"""
Example of a model fitting script.
The user should modify the model
according to the characteristics of
the signal of interest.
Part of the code is taken from the 
kmpfit examples:
http://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#profile-fitting
"""

import numpy as np
import argparse
from lmfit import Model
from scipy.special import wofz
from crrlpy import crrls
ln2 = np.log(2)

def voigt(x, y):
    # The Voigt function is also the real part of 
    # w(z) = exp(-z^2) erfc(iz), the complex probability function,
    # which is also known as the Faddeeva function. Scipy has 
    # implemented this function under the name wofz()

    z = x + 1j*y
    I = wofz(z).real

    return I

def Voigt(nu, alphaD, alphaL, nu_0, A, a, b):
    """
    The Voigt line shape in terms of its physical parameters
    nu: independent variable
    alphaD: FWHM of the Gaussian
    alphaL: FWHM of the Lorentzian
    nu_0: the line center
    A: the line area
    a, b: background parameters. bkgd = a + b*nu
    """

    f = np.sqrt(ln2)
    x = (nu - nu_0)/alphaD * f
    y = alphaL/alphaD * f
    bkgd = a + b*nu 
    V = A*f/(alphaD*np.sqrt(np.pi)) * voigt(x, y) + bkgd

    return V

def funcV(x, p):
    # Compose the Voigt line-shape
    
    alphaD, alphaL, nu_0, I, a, b = p
    
    return Voigt(x, alphaD, alphaL, nu_0, I, a, b)

def main(spec, output, plot):
    """
    """
    
    dD = 3 # 3 km/s Doppler FWHM for the lines
    
    data = np.loadtxt(spec)
    x = data[:,0]
    y = data[:,1]
    w = data[:,2]
    
    # Catch NaNs and invalid values:
    mask_x = np.ma.masked_equal(x, -9999).mask
    mask_y = np.isnan(y)
    mask = np.array(reduce(np.logical_or, [mask_x, mask_y]))
    
    mx = x[~mask]
    my = y[~mask]
    mw = w[~mask]
    
    # Create the model and set the parameters
    mod1 = Model(Voigt, prefix='V1_')
    pars = mod1.make_params()
    mod2 = Model(Voigt, prefix='V2_')
    pars += mod2.make_params()
    mod = mod1 + mod2
    
    # Edit the model parameter starting values, conditions, etc...
    # Background parameters
    pars['V1_a'].set(value=0, expr='', vary=False)
    pars['V1_b'].set(value=0, expr='', vary=False)
    pars['V2_a'].set(value=0, expr='', vary=False)
    pars['V2_b'].set(value=0, expr='', vary=False)
    # Line center
    pars['V1_nu_0'].set(value=-47., vary=True, min=-50, max=-44)
    pars['V2_nu_0'].set(value=-38., vary=True, min=-40, max=-36)
    # Line area
    pars['V1_A'].set(value=-1e-2, max=-1e-8)
    pars['V2_A'].set(value=-1e-2, max=-1e-8)
    # Line width
    pars['V1_alphaD'].set(value=dD, vary=False)
    pars['V2_alphaD'].set(value=dD, vary=False)
    pars['V1_alphaL'].set(value=1, vary=True, min=0)
    pars['V2_alphaL'].set(value=1, vary=True, min=0)
    
    # Fit the model using a weight
    #print len(my), len(mx)
    modx = np.array([mx, mx, mx])
    
    try:
        fit = mod.fit(my, pars, nu=mx, weights=mw)
        fit1 = Voigt(mx, fit.params['V1_alphaD'].value, fit.params['V1_alphaL'].value, 
                     fit.params['V1_nu_0'].value, fit.params['V1_A'].value, 
                     fit.params['V1_a'].value, 0)
        fit2 = Voigt(mx, fit.params['V2_alphaD'].value, fit.params['V2_alphaL'].value, 
                     fit.params['V2_nu_0'].value, fit.params['V2_A'].value, 0, 0)
        fit3 = fit.best_fit
        mody = np.array([fit1, fit2, fit3])
        bfit = fit.best_fit
    except TypeError:
        bfit = np.zeros(mx.shape)
        mody = np.zeros(modx.shape)
        
    
    
    #fit1 = Voigt(mx, fit.params['V1_alphaD'].value, fit.params['V1_alphaL'].value, 
                 #fit.params['V1_nu_0'].value, fit.params['V1_A'].value, fit.params['V1_a'].value, 0)
    #fit2 = Voigt(mx, fit.params['V2_alphaD'].value, fit.params['V2_alphaL'].value, 
                 #fit.params['V2_nu_0'].value, fit.params['V2_A'].value, 0, 0)
    #fit3 = fit.best_fit
    #mody = np.array([fit1, fit2, fit3])
    #modx = np.array([mx, mx, mx])
    
    if len(mx) == 0:
        mx = np.zeros(10)
        my = np.zeros(10)
        bfit = np.zeros(10)
        modx = np.array([mx, mx, mx])
        mody = np.zeros(modx.shape)
    
    crrls.plot_model(mx, my, modx, mody, plot)
    
    np.savetxt(output, np.c_[mx, bfit])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', type=str,
                        help="Spectrum to be fit. (string)")
    parser.add_argument('output', type=str,
                        help="Name of the output file with the best fit model. (string)")
    parser.add_argument('plot', type=str,
                        help="Name of the output figure. (string)")
    args = parser.parse_args()
    
    main(args.spec, args.output, args.plot)