#!/usr/bin/env python
from __future__ import division

import itertools
import os
import re
import collections

import matplotlib as mpl
havedisplay = "DISPLAY" in os.environ
if not havedisplay:
    mpl.use('Agg')
import pylab as plt
import numpy as np
import scipy.integrate as sint

from astropy.constants import k_B
from lmfit import Model
from lmfit.models import VoigtModel, ConstantModel, GaussianModel, PolynomialModel, PowerLawModel
from scipy.special import wofz
from scipy import interpolate
from scipy.constants import c
from scipy.signal import wiener#, savgol_filter
from scipy.ndimage.filters import gaussian_filter
from frec_calc import set_dn, make_line_list

def alphanum_key(s):
    """ 
    Turn a string into a list of string and number chunks.
    
    :param s: String
    :returns: List with strings and integers.
    :rtype: list
    
    :Example:
    
    >>> alphanum_key("z23a")
    ["z", 23, "a"]
    
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def average(data, axis, n):
    """
    Averages data along the given axis by combining n adjacent values.
    
    :param data: Data to average along a given axis.
    :type data: numpy array
    :param axis: Axis along which to average.
    :type axis: int
    :param n: Factor by which to average.
    :type n: int
    :returns: Data decimated by a factor n along the given axis.
    :rtype: numpy array
    """
    
    if n < 1:
        print "Will not work"
        avg_tmp = data
    else:
        avg_tmp = 0
        for i in xrange(n):
            si = (n - 1) - i
            if si <= 0:
                avg_tmp += data[i::n]
            else:
                avg_tmp += data[i:-si:n]
        avg_tmp = avg_tmp/n
        
        return avg_tmp

def best_match_indx(value, array, tol):
    """
    Searchs for the best match to a value inside an array given a tolerance.
    
    :param value: Value to find inside the array.
    :type value: float
    :param tol: Tolerance for match.
    :type tol: float
    :param array: List to search for the given value.
    :type array: numpy.array
    :return: Best match for val inside array.
    :rtype: float
    """
    
    upp = np.where(array >= value - tol)[0]
    low = np.where(array <= value + tol)[0]
    
    if bool(set(upp) & set(low)):
        out = iter(set(upp) & set(low)).next()
        #return iter(set(upp) & set(low)).next()
    elif low.any():
        out = low[-1]
        #return low[-1]
    else:
        out = upp[0]
        #return upp[0]
    return out
    
def best_match_indx2(value, array):
    """
    Searchs for the index of the closest entry to value inside an array.
    
    :param value: Value to find inside the array.
    :type value: float
    :param array: List to search for the given value.
    :type array: list or numpy.array
    :return: Best match index for the value inside array.
    :rtype: float
    
    :Example:
    
    >>> a = [1,2,3,4]
    >>> best_match_indx2(3, a)
    2
    
    """
    
    array = np.array(array)
    subarr = abs(array - value)
    subarrmin = subarr.min()
    
    return np.where(subarr == subarrmin)[0][0]

def best_match_value(value, array):
    """
    Searchs for the closest ocurrence of value in array.
    
    :param value: Value to find inside the array.
    :type value: float
    :param array: List to search for the given value.
    :type array: list or numpy.array
    :return: Best match for the value inside array.
    :rtype: float.
    
    :Example:
    
    >>> a = [1,2,3,4]
    >>> best_match_value(3.5, a)
    3
    
    """
    
    array = np.array(array)
    subarr = abs(array - value)
    subarrmin = subarr.min()
    
    return array[np.where(subarr == subarrmin)[0][0]]
    
def blank_lines(freq, tau, reffreqs, v0, dv0):
    """
    Blanks the lines in a spectra.
    
    :param freq: Frequency axis of the spectra.
    :type freq: array
    :
    """
    
    try:
        for ref in reffreqs:
            #print "#freq: {0} #tau: {1}".format(len(freq), len(tau))
            lm0, lmf = get_line_mask(freq, ref, v0, dv0)
            freq = np.concatenate((freq[:lm0], freq[lmf:]))
            tau = np.concatenate((tau[:lm0], tau[lmf:]))
    except TypeError:
        lm0, lmf = get_line_mask(freq, reffreqs, v0, dv0)
        freq = np.concatenate((freq[:lm0], freq[lmf:]))
        tau = np.concatenate((tau[:lm0], tau[lmf:]))
        
    return freq, tau

def blank_lines2(freq, tau, reffreqs, dv):
    """
    """
    try:
        for ref in reffreqs:
            #print "#freq: {0} #tau: {1}".format(len(freq), len(tau))
            lm0, lmf = get_line_mask2(freq, ref, dv)
            freq = np.concatenate((freq[:lm0], freq[lmf:]))
            tau = np.concatenate((tau[:lm0], tau[lmf:]))
    except TypeError:
        lm0, lmf = get_line_mask2(freq, reffreqs, dv)
        freq = np.concatenate((freq[:lm0], freq[lmf:]))
        tau = np.concatenate((tau[:lm0], tau[lmf:]))
        
    return freq, tau

def df2dv(f0, df):
    """
    Convert a frequency delta to a velocity delta given a central frequency.
    
    :param f0: Rest frequency. (Hz)
    :type f0: float
    :param df: Frequency delta. (Hz)
    :type df: float
    :returns: The equivalent velocity delta for the given frequency delta.
    :rtype: float in Hz
    """
    
    return c*(df/f0)

def dv2df(f0, dv):
    """
    Convert a velocity delta to a frequency delta given a central frequency.
    
    :param f0: Rest frequency. (Hz)
    :type f0: float
    :param dv: Velocity delta. (m/s)
    :type dv: float
    :returns: The equivalent frequency delta for the given velocity delta.
    :rtype: float in :math:`\mbox{m s}^{-1}`
    """
    
    return dv*f0/c

def dv_minus_doppler(dV, ddV, dD, ddD):
    """
    Returns the Lorentzian contribution to the line width assuming that the line has a Voigt profile.
    
    :param dV: Total line width
    :type dV: float
    :param ddV: Uncertainty in the total line width.
    :type ddV: float
    :param dD: Doppler contribution to the line width.
    :type dD: float
    :param ddD: Uncertainty in the Doppler contribution to the line width.
    :returns: The Lorentz contribution to the total line width.
    :rtype: float
    """
    
    a = 0.5346
    b = 0.2166
    
    d = np.power(2.*a*dV, 2) - 4.*(b - a*a)*(np.power(dD, 2.) - np.power(dV, 2.))
    
    if d < 0:
        print "No real solutions, check input."
        return 0
    
    dL_p = (-2.*a*dV + np.sqrt(d))/(2.*(b - a*a))
    dL_m = (-2.*a*dV - np.sqrt(d))/(2.*(b - a*a))
    
    if dL_m < dV:
        dL = dL_m
        ddL1 = (-2.*a - ((a*a*dV) + 8.*(b - a*a)*dV)/np.sqrt(np.power(2.*a*dV, 2.) - 4.*(b-a*a)*(np.power(dD, 2.) - np.power(dV, 2.))))/(2.*(b - a*a))
    else:
        dL = dL_p
        ddL1 = (-2.*a + ((a*a*dV) + 8.*(b - a*a)*dV)/np.sqrt(np.power(2.*a*dV, 2.) - 4.*(b-a*a)*(np.power(dD, 2.) - np.power(dV, 2.))))/(2.*(b - a*a))
        
    ddL2 = 4.*(b - a*a)*dD/np.sqrt(np.power(2.*a*dV, 2.) - 4.*(b - a*a)*(np.power(dD, 2.) - np.power(dV, 2.)))
    
    ddL = np.sqrt(np.power(ddL1*ddV, 2) + np.power(ddL2*ddV, 2)) 
    
    return dL, ddL

def dv_minus_doppler2(dV, ddV, dD, ddD):
    """
    Returns the Lorentzian contribution to the line width assuming that the line has a Voigt profile.
    
    :param dV: Total line width
    :type dV: float
    :param ddV: Uncertainty in the total line width.
    :type ddV: float
    :param dD: Doppler contribution to the line width.
    :type dD: float
    :param ddD: Uncertainty in the Doppler contribution to the line width.
    :returns: The Lorentz contribution to the total line width.
    :rtype: float
    """
    
    a = 0.5346
    b = 0.2166
    
    den = (a*a - b)
    dif = np.power(dV, 2.) - np.power(dD, 2.)
    d = np.power(a*dV, 2) - den*dif
    
    if d < 0:
        print "No real solutions, check input."
        return 0
    
    dL_p = (a*dV + np.sqrt(d))/den
    dL_m = (a*dV - np.sqrt(d))/den
    
    if dL_m < dV:
        dL = dL_m
        ddL1 = (a + (a*a*dV - dV*den)/np.sqrt(np.power(a*dV, 2) - den*dif))/den
    else:
        dL = dL_p
        ddL1 = (a - (a*a*dV - dV*den)/np.sqrt(np.power(a*dV, 2) - den*dif))/den
        
    ddL2 = dD/np.sqrt(np.power(a*dV, 2) - den*dif)
    
    ddL = np.sqrt(np.power(ddL1*ddV, 2) + np.power(ddL2*ddV, 2)) 
    
    return dL, ddL

def f2n(f, line, n_max=1500):
    """
    Converts a given frequency to a principal quantum number n for a given line.
    """
    
    line, nn, freq = make_line_list(line, n_max=n_max)
    fii = np.in1d(freq, f)
    
    return nn[fii]

def find_lines_in_band(freq, species='CI', transition='alpha', z=0, verbose=False):
    """
    Finds if there are any lines corresponding to transitions of the given species in the frequency range. \
    The line transition frequencies are corrected for redshift.
    
    :param freq:
    :param species:
    :param z: Redshift to apply to the rest frequencies.
    :param verbose: Verbose output?
    :type verbose: bool
    :returns:
    :rtype: List of principal quantum numbers and list of reference frequencies.
    """
    
    # Load the reference frequencies
    qn, restfreq = load_ref(species, transition)
    
    # Correct rest frequencies for redshift
    reffreq = restfreq/(1.0 + z)
    
    # Check which lines lie within the subband
    mask_ref = (freq[0] < reffreq) & (freq[-1] > reffreq)
    reffreqs = reffreq[mask_ref]
    refqns = qn[mask_ref]
    
    nlin = len(reffreqs)
    if verbose:
        print "Found {0} {1} lines within the subband.".format(nlin, transition)
        if nlin > 1:
            print "Corresponding to n values: {0}--{1}".format(refqns[0], refqns[-1])
        elif nlin == 1:
            print "Corresponding to n value {0} and frequency {1} MHz".format(refqns[0], reffreqs[0])
    
    return refqns, reffreqs

def find_lines_sb(freq, transition, z=0, verbose=False):
    """
    Finds if there are any lines corresponding to 
    transitions of the given species in the frequency range.
    The line transition frequencies are corrected for redshift.
    """
    
    # Load the reference frequencies
    qn, restfreq = load_ref2(transition)
    
    # Correct rest frequencies for redshift
    reffreq = restfreq/(1.0 + z)
    
    # Check which lines lie within the subband
    mask_ref = (freq[0] < reffreq) & (freq[-1] > reffreq)
    reffreqs = reffreq[mask_ref]
    refqns = qn[mask_ref]
    
    nlin = len(reffreqs)
    if verbose:
        print "Found {0} {1} lines within the subband.".format(nlin, transition)
        if nlin > 1:
            print "Corresponding to n values: {0}--{1}".format(refqns[0], refqns[-1])
        elif nlin == 1:
            print "Corresponding to n value {0} and frequency {1} MHz".format(refqns[0], reffreqs[0])

    return refqns, reffreqs

def fit_continuum(x, y, degree, p0):
    """
    Divide tb by given a model and starting parameters p0.
    Returns: tb/model - 1
    """
    
    # Divide by linear baseline
    mod = PolynomialModel(degree)
    params = mod.make_params()
    if len(p0) != len(params):
        print "Insuficient starting parameter values."
        return 0
    else:
        for param in params:
            params[param].set(value=p0[param])

    fit = mod.fit(y, x=x, params=params)
    
    return fit

def fit_line(sb, n, ref, vel, tau, rms, model, v0=None, verbose=True):
    
    # Set the model used to fit the line
    #mod, nparams = set_fit_model(model)
    
    # Set up dictionary with fit results
    params = fit_storage()
    
    params['n'] = n
    params['sb'] = sb
    params['reffreq'] = ref
    params['rms'] = rms
    
    # Setup initial fit parameters
    # Peak
    tmin = tau.min() 
    peak_indx = np.where(tau == tmin)[0]
    if len(peak_indx) > 1:
        peak_indx = peak_indx[0]
        
    # Peak velocity
    if v0:
        vel0 = v0
    else:
        vel0 = vel[peak_indx] 
    # Line width
    # Choose the minimum between 2 options for the line width
    if peak_indx != 0:
        dv1 = abs(vel[peak_indx] - vel[peak_indx-1])
    else:
        dv1 = 10
    if peak_indx != len(vel) - 1:
        dv2 = abs(vel[peak_indx] - vel[peak_indx+1])
    else:
        dv2 = 10
    sx0 = min(dv1, dv2)
    # Use the rms as optical depth offset from 0
    A0 = np.std(tau, ddof=1)
    #np.mean([tbl[fit['ch0'][i]:fit['ch0'][i]+10].mean(), tbl[fit['chf'][i]-10:fit['chf'][i]].mean()])
    
    npts = len(tau)
    if verbose:
        print "p0: ", tmin,vel0,sx0,A0
        print "# unmasked points to fit: {0}".format(npts)
        
    #try:
    #if npts > nparams:
    if model == 'gauss':
        gmod = GaussianModel()
        cmod = ConstantModel()
        pars = cmod.guess(tau, x=vel)
        pars += gmod.make_params(amplitude=tmin, center=vel0, sigma=sx0)
        pars['center'].set(min=-60, max=-40)
        pars['amplitude'].set(value=tmin, max=0, min=-0.1)
        mod = gmod + cmod
        fit = mod.fit(tau, pars, x=vel)
    elif model == 'gauss0':
        mod = GaussianModel()
        pars = mod.guess(tau, x=vel)
        pars['center'].set(value=vel0, min=-60, max=-40)
        pars['amplitude'].set(value=tmin, max=0, min=-0.1)
        fit = mod.fit(tau, pars, x=vel)
    elif model == '2gauss':
        gmod1 = GaussianModel(prefix='g1_')
        gmod2 = GaussianModel(prefix='g2_')
        cmod = ConstantModel()
        pars = cmod.guess(tau, x=vel)
        pars += gmod1.make_params()
        pars += gmod2.make_params()
        mod = gmod1 + gmod2 + cmod
        pars['g1_center'].set(value=vel0, min=-60, max=-40)
        pars['g2_center'].set(value= vel0+9.4, min=-60, max=-40, expr='g1_center+9.4')
        pars['g1_amplitude'].set(value=tmin, max=0)
        pars['g2_amplitude'].set(value=tmin, max=0)
        fit = mod.fit(tau, pars, x=vel)
    elif model == 'voigt0':
        vmod = VoigtModel()
        pars = vmod.make_params(amplitude=tmin, center=vel0, sigma=3)
        mod = vmod
        pars['gamma'].set(value=3, vary=True, expr='', min=0.0)
        pars['center'].set(value=vel0, vary=False)
        pars['amplitude'].set(max=0.0)
        pars['sigma'].set(value=FWHM2sigma(3), vary=False)
        fit = mod.fit(tau, pars, x=vel)
    elif model == 'voigt':
        vmod = VoigtModel()
        cmod = ConstantModel()
        pars = cmod.guess(tau, x=vel)
        pars += vmod.make_params(amplitude=tmin, center=vel0, sigma=3)
        mod = vmod + cmod
        pars['gamma'].set(value=100, vary=True, expr='', min=0.0)
        #pars['gamma'].set(min=0.001, max=10000)
        #pars['center'].set(min=-60., max=-35.)
        pars['center'].set(value=vel0, vary=True)
        pars['amplitude'].set(max=0.0)
        pars['sigma'].set(value=FWHM2sigma(3), vary=False)
        fit = mod.fit(tau, pars, x=vel)
    elif model == 'voigt2':
        vmod1 = VoigtModel(prefix='v1_')
        vmod2 = VoigtModel(prefix='v2_')
        cmod = ConstantModel()
        pars = cmod.guess(tau, x=vel)
        pars += vmod1.make_params()
        pars += vmod2.make_params()
        mod = vmod1 + vmod2 + cmod
        pars['v1_gamma'].set(value=3, vary=True, expr='', min=0.0)
        pars['v2_gamma'].set(value=3, vary=True, expr='', min=0.0)
        pars['v1_center'].set(value=vel0, vary=True)
        pars['v2_center'].set(value=vel0+9.4, expr='v1_center+9.4', vary=False)
        pars['v1_amplitude'].set(max=0.0, value=tmin)
        pars['v2_amplitude'].set(max=0.0, value=0.5*tmin)
        pars['v1_sigma'].set(value=FWHM2sigma(3), vary=False)
        pars['v2_sigma'].set(value=FWHM2sigma(3), vary=False)
        fit = mod.fit(tau, pars, x=vel)
        
    best_fit = fit.best_fit
    
    # Correct LMfit model definitions
    if model == 'gauss':
        # Line center
        params['vp1'] = fit.params['center'].value   # Velocity of the peak
        params['dvp1'] = fit.params['center'].stderr
        # Line width
        params['dv1'] = fit.params['fwhm'].value     # Line width in velocity
        params['ddv1'] = fit.params['fwhm'].stderr
        # Peak optical depth
        params['tau1'] = fit.params['amplitude'].value / \
                        (np.sqrt(2*np.pi) * fit.params['sigma'].value)
        params['dtau1'] = fit.params['amplitude'].stderr / \
                            (np.sqrt(2*np.pi) * fit.params['sigma'].value)
        # Assume Gaussian profile for integrated optical depth
        params['itau1'] = params['tau1']*params['dv1']*np.sqrt(np.pi/np.log(16)) 
        ditau1 = np.power(params['itau1']/params['tau1']*params['dtau1'], 2)
        ditau2 = np.power(params['itau1']/params['dv1']*params['ddv1'], 2)
        params['ditau1'] = np.sqrt(ditau1 + ditau2)
    
    elif model == 'voigt' or model == 'voigt0':
        # Line center
        params['vp1'] = fit.params['center'].value   # Velocity of the peak
        params['dvp1'] = fit.params['center'].stderr
        # Line width
        #params['dv1'] = fit.params['fwhm'].value     # Line width in velocity
        #params['ddv1'] = fit.params['fwhm'].stderr
        sigma = fit.params['sigma'].value
        dsigma = fit.params['sigma'].stderr
        dD = sigma2FWHM(sigma)
        ddD = sigma2FWHM(dsigma)
        gamma = fit.params['gamma'].value
        dgamma = fit.params['gamma'].stderr
        dL = 2*gamma
        ddL = 2*dgamma
        params['dv1'] = line_width(dD, dL)
        params['ddv1'] = line_width_err(dD, dL, ddD, ddL)
        
        # Line amplitude
        params['ptau'] = min(best_fit)
        params['tau1'] = fit.params['amplitude'].value / \
                        (np.sqrt(2*np.pi) * fit.params['sigma'].value)
        params['dtau1'] = fit.params['amplitude'].stderr / \
                         (np.sqrt(2*np.pi) * fit.params['sigma'].value)
        # Line area
        params['itau1'] = voigt_area(params['ptau'], params['dv1'], 
                                     fit.params['gamma'].value, 
                                     fit.params['sigma'].value)
        params['ditau1'] = voigt_area_err(params['itau1'], 
                                          params['ptau'], 
                                          params['dtau1'], 
                                          params['dv1'], 
                                          params['ddv1'], 
                                          fit.params['gamma'].value, 
                                          fit.params['sigma'].value)
        params['dL'] = fit.params['gamma'].value
        params['ddL'] = fit.params['gamma'].stderr
        params['dD'] = fit.params['sigma'].value
        params['ddD'] = fit.params['sigma'].stderr
    elif model == 'voigt2':
        # Line center
        params['vp1'] = fit.params['v1_center'].value   # Velocity of the peak
        params['dvp1'] = fit.params['v1_center'].stderr
        params['vp2'] = fit.params['v2_center'].value   # Velocity of the peak
        params['dvp2'] = fit.params['v2_center'].stderr
        # Line width
        #params['dv1'] = fit.params['fwhm'].value     # Line width in velocity
        #params['ddv1'] = fit.params['fwhm'].stderr
        # Line width
        sigma = fit.params['v1_sigma'].value
        dsigma = fit.params['v1_sigma'].stderr
        dD = sigma2FWHM(sigma)
        ddD = sigma2FWHM(dsigma)
        gamma = fit.params['v1_gamma'].value
        dgamma = fit.params['v1_gamma'].stderr
        dL = 2*gamma
        ddL = 2*dgamma
        params['dv1'] = line_width(dD, dL)
        params['ddv1'] = line_width_err(dD, dL, ddD, ddL)
        
        # Line width
        sigma = fit.params['v2_sigma'].value
        dsigma = fit.params['v2_sigma'].stderr
        dD = sigma2FWHM(sigma)
        ddD = sigma2FWHM(dsigma)
        gamma = fit.params['v2_gamma'].value
        dgamma = fit.params['v2_gamma'].stderr
        dL = 2*gamma
        ddL = 2*dgamma
        params['dv2'] = line_width(dD, dL)
        params['ddv2'] = line_width_err(dD, dL, ddD, ddL)
        
        # Line amplitude
        params['ptau'] = min(best_fit)
        params['tau1'] = fit.params['v1_amplitude'].value / \
                        (np.sqrt(2*np.pi) * fit.params['v1_sigma'].value)
        params['dtau1'] = fit.params['v1_amplitude'].stderr / \
                         (np.sqrt(2*np.pi) * fit.params['v1_sigma'].value)
        params['tau2'] = fit.params['v2_amplitude'].value / \
                        (np.sqrt(2*np.pi) * fit.params['v2_sigma'].value)
        params['dtau2'] = fit.params['v2_amplitude'].stderr / \
                         (np.sqrt(2*np.pi) * fit.params['v2_sigma'].value)
        # Line area
        params['itau1'] = voigt_area(params['tau1'], params['dv1'], 
                                     fit.params['v1_gamma'].value, 
                                     fit.params['v1_sigma'].value)
        params['ditau1'] = voigt_area_err(params['itau1'], 
                                          params['tau1'], 
                                          params['dtau1'], 
                                          params['dv1'], 
                                          params['ddv1'], 
                                          fit.params['v1_gamma'].value, 
                                          fit.params['v1_sigma'].value)
        params['itau2'] = voigt_area(params['tau2'], params['dv2'], 
                                     fit.params['v2_gamma'].value, 
                                     fit.params['v2_sigma'].value)
        params['ditau2'] = voigt_area_err(params['itau2'], 
                                          params['tau2'], 
                                          params['dtau2'], 
                                          params['dv2'], 
                                          params['ddv2'], 
                                          fit.params['v2_gamma'].value, 
                                          fit.params['v2_sigma'].value)
        params['dL'] = fit.params['v1_gamma'].value
        params['ddL'] = fit.params['v1_gamma'].stderr
        params['dD'] = fit.params['v1_sigma'].value
        params['ddD'] = fit.params['v1_sigma'].stderr
        params['dL2'] = fit.params['v2_gamma'].value
        params['ddL2'] = fit.params['v2_gamma'].stderr
        params['dD2'] = fit.params['v2_sigma'].value
        params['ddD2'] = fit.params['v2_sigma'].stderr
        
    elif model == '2gauss':
        # Peak optical depth
        params['tau1'] = fit.params['g1_amplitude'].value / \
                        (np.sqrt(2*np.pi) * fit.params['g1_sigma'].value)
        params['dtau1'] = fit.params['g1_amplitude'].stderr / \
                            (np.sqrt(2*np.pi) * fit.params['g1_sigma'].value)
        params['tau2'] = fit.params['g2_amplitude'].value / \
                        (np.sqrt(2*np.pi) * fit.params['g2_sigma'].value)
        params['dtau2'] = fit.params['g2_amplitude'].stderr / \
                            (np.sqrt(2*np.pi) * fit.params['g2_sigma'].value)
        params['vp1'] = fit.params['g1_center'].value   # Velocity of the peak
        params['dvp1'] = fit.params['g1_center'].stderr
        params['dv1'] = fit.params['g1_fwhm'].value     # Line width in velocity
        params['ddv1'] = fit.params['g1_fwhm'].stderr
        params['vp2'] = fit.params['g2_center'].value   # Velocity of the peak
        params['dvp2'] = fit.params['g2_center'].stderr
        params['dv2'] = fit.params['g2_fwhm'].value     # Line width in velocity
        params['ddv2'] = fit.params['g2_fwhm'].stderr
        params['tau1'] = fit.params['g1_amplitude'].value / \
                        (np.sqrt(2*np.pi) * fit.params['g1_sigma'].value)
        params['dtau1'] = fit.params['g1_amplitude'].stderr / \
                            (np.sqrt(2*np.pi) * fit.params['g1_sigma'].value)
        params['tau2'] = fit.params['g2_amplitude'].value / \
                        (np.sqrt(2*np.pi) * fit.params['g2_sigma'].value)
        params['dtau2'] = fit.params['g2_amplitude'].stderr / \
                            (np.sqrt(2*np.pi) * fit.params['g2_sigma'].value)
        params['itau1'] = params['tau1']*params['dv1']*np.sqrt(np.pi/np.log(16)) 
        ditau1 = np.power(params['itau1']/params['tau1']*params['dtau1'], 2)
        ditau2 = np.power(params['itau1']/params['dv1']*params['ddv1'], 2)
        params['ditau1'] = np.sqrt(ditau1 + ditau2)
        params['itau2'] = params['tau2']*params['dv2']*np.sqrt(np.pi/np.log(16)) 
        ditau1 = np.power(params['itau2']/params['tau2']*params['dtau2'], 2)
        ditau2 = np.power(params['itau2']/params['dv2']*params['ddv2'], 2)
        params['ditau2'] = np.sqrt(ditau1 + ditau2)
    #print model
    if '0' not in model:
        params['tau0'] = fit.params['c'].value      # Spectrum constant offset from 0
        params['dtau0'] = fit.params['c'].stderr
    
    else:
        params['tau0'] = 0
        params['dtau0'] = rms
    
        
    # Godness of fit
    params['chi2'] = fit.chisqr
    params['chiv2'] = fit.redchi
    params['res'] = np.mean(fit.residual)
        
    
    if verbose:
        print "Velocity fit results"
        print fit.fit_report()
        print ("Integrated optical depth:"),
        print ("{0} +/- {1} km/s".format(params['itau1'],
                                            params['ditau1']))
    
    return best_fit, params, fit.residual, fit

def fit_storage():
    """
    Returns a dictionary with the 
    entries for the parameters to 
    be fitted.
    """
    # Fit results
    #if model == gaussian
    blankval = -99
    fit = collections.OrderedDict((('sb',blankval),       #0 Sub Band number
                                   ('n',blankval),        #1 Principal quantum number
                                   ('reffreq',blankval),  #2 Reference frequency
                                   ('tau1',blankval),     #3 Peak optical depth
                                   ('vp1',blankval),      #4 Velocity of peak optical depth
                                   ('dv1',blankval),      #5 Line width
                                   ('itau1',blankval),    #6 Integrated optical depth
                                   ('tau0',blankval),     #7 Optical depth offset
                                   ('dtau1',blankval),    #8 Peak optical depth error
                                   ('dvp1',blankval),     #9 Velocity of peak optical depth error
                                   ('ddv1',blankval),     #10 Line width error
                                   ('ditau1',blankval),   #11 Integrated optical depth error
                                   ('dtau0',blankval),    #12 Optical depth offset error
                                   ('dD',blankval),       #13
                                   ('ddD',blankval),      #14
                                   ('dL',blankval),       #15
                                   ('ddL',blankval),      #16
                                   ('chiv2',blankval),    #17 Reduced Chi squared
                                   ('chi2',blankval),     #18 Chi squared
                                   ('res',blankval),      #19 Fit residuals
                                   ('rms',blankval),      #20 Continuum rms
                                   ('pres',blankval),     #21 Peak residual
                                   ('flag',blankval),     #22 Fit result flag
                                   ('tau2',blankval),     #23
                                   ('vp2',blankval),      #24
                                   ('dv2',blankval),      #25
                                   ('itau2',blankval),    #26
                                   ('dtau2',blankval),    #27
                                   ('dvp2',blankval),     #28
                                   ('ddv2',blankval),     #29
                                   ('ditau2',blankval),   #30
                                   ('dD2',blankval),      #31
                                   ('ddD2',blankval),     #32
                                   ('dL2',blankval),      #33
                                   ('ddL2',blankval),     #34
                                   ('ptau',blankval)))    #35
    

    return fit

def freq2vel(f0, f):
    """
    Convert a frequency axis to a velocity axis given a central frequency.
    Uses the radio definition of velocity.
    
    :param f0: Rest frequency for the conversion. (Hz)
    :type f0: float
    :param f: Frequencies to be converted to velocity. (Hz)
    :type f: numpy array
    :returns: f converted to velocity given a rest frequency :math:`f_{0}`.
    :rtype: numpy array
    """
    
    return c*(1. - f/f0)

def FWHM2sigma(fwhm):
    """
    Converts a FWHM to the standard deviation of a Gaussian distribution.
    """
    
    return fwhm/(2.*np.sqrt(2.*np.log(2.)))

def Gauss(y, **kwargs):
    """
    Applies a Gaussian filter to y.
    """
    
    gauss = gaussian_filter(y, sigma=kwargs['sigma'], order=kwargs['order'])
    
    return gauss

def gauss_area(amplitude, sigma):
    """
    Returns the area under a Gaussian of a given amplitude and sigma.
    """
    
    return amplitude*sigma*np.sqrt(2*np.pi)

def gauss_area_err(amplitude, amplitude_err, sigma, sigma_err):
    """
    """
    
    err1 = np.power(amplitude_err*sigma*np.sqrt(2*np.pi), 2)
    err2 = np.power(sigma_err*amplitude*np.sqrt(2*np.pi), 2)
    
    return np.sqrt(err1 + err2)

def Gaussian(x, sigma, center, amplitude):
    """
    1-d Gaussian with no amplitude offset.
    """
    
    return amplitude*np.exp(-np.power((x - center), 2.)/(2.*np.power(sigma, 2.)))

def gaussian_off(x, amplitude, center, sigma, c):
    """
    1-d Gaussian with a constant amplitude offset.
    """
    
    return amplitude*np.exp(-np.power((x - center), 2.)/(2.*sigma**2.)) + c

def get_axis(header, axis):
    """
    Constructs a cube axis
    
    :param header: Fits cube header.
    :type header: pyfits header
    :param axis: Axis to reconstruct.
    :type axis: int
    :returns: cube axis
    :rtype: numpy array
    """
    
    axis = str(axis)
    dx = header.get("CDELT" + axis)
    try:
        dx = float(dx)
        p0 = header.get("CRPIX" + axis)
        x0 = header.get("CRVAL" + axis)
        
    except TypeError:
        dx = 1
        p0 = 1
        x0 = 1

    n = header.get("NAXIS" + axis)
    
    p0 -= 1 # Fits files index start at 1, not for python.
    
    return np.arange(x0 - p0*dx, x0 - p0*dx + n*dx, dx)

def get_line_mask(freq, reffreq, v0, dv0):
    """
    Return a mask with ranges where a line is expected in the given frequency range for \
    a line with a given reference frequency at expected velocity v0 and line width dv0.
    
    :param freq: Frequency axis where the line is located.
    :type freq: numpy array or list
    :param reffreq: Reference frequency for the line.
    :type reffreq: float
    :param v0: Velocity of the line. (km/s)
    :type v0: float
    :param dv0: Velocity range to mask. (km/s)
    :type dv0: float
    :returns: Mask centered at the line center and width dv0 referenced to the input :paramref:`freq`.
    """
    
    #print v0
    f0 = vel2freq(reffreq, v0*1e3)
    #print "Line location in frequency {0} MHz".format(f0)
    df0 = dv2df(reffreq*1e6, dv0*1e3)
    #print "Line width in frequency {0} Hz".format(df0)
    
    df = abs(freq[0] - freq[1])
    #print "Channel width {} Hz".format(df*1e6)
    
    f0_indx = best_match_indx(f0, freq, df/2.0)
    #f0_indx = abs(freq - f0).argmin()
    #print "Frequency index: {}".format(f0_indx)
    
    #print "Line width in channels: {}".format(df0/df/1e6)
    
    mindx0 = f0_indx - df0/df/1e6
    mindxf = f0_indx + df0/df/1e6
    
    #print mindx0, mindxf
    return [mindx0, mindxf]

def get_line_mask2(freq, reffreq, dv):
    """
    Return a mask with ranges where a line is expected in the given frequency range for \
    a line with a given reference frequency and line width dv.
    
    
    """
    
    df = dv2df(reffreq, dv*1e3)
    df_chan = get_min_sep(freq)
    f0_indx = best_match_indx2(reffreq, freq)

    f_mini = f0_indx - df/df_chan
    if f_mini < 0:
        f_mini = 0
    f_maxi = f0_indx + df/df_chan

    return [f_mini, f_maxi]

def get_rms(data, axis=None):
    """
    Computes the rms of the given data.
    
    :param data: Array with values where to compute the rms.
    :type data: numpy array or list
    :param axis: Axis over which to compute the rms. Default: None
    :type axis: int
    :returns: The rms of data.
    .. math::
    
        \\mbox{\\it rms}=\\sqrt{\\langle\\mbox{data}\\rangle^{2}+V[\\mbox{data}]}
        
    where :math:`V` is the variance of the data.
    """
    rms = np.sqrt(np.power(np.std(data, axis=axis), 2) \
           + np.power(np.mean(data, axis=axis), 2))
    rms = np.sqrt(np.average(np.power(data, 2)))
    return rms

def get_min_sep(array):
    """
    Get the minimum element separation in
    an array.
    """

    da = min(abs(array[0:-1:2] - array[1::2]))
    return da

def is_number(s):
    """
    Checks wether a string is a number or not.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def linear(x, a, b):
    """
    Linear model.
    """
    return a*x + b

def line_width(dD, dL):
    """
    http://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile
    """
    
    return np.multiply(0.5346, dL) + np.sqrt(np.multiply(0.2166, np.power(dL, 2)) + np.power(dD, 2))

def line_width_err(dD, dL, ddD, ddL):
    """
    Computes the error in the FWHM of
    a Voigt profile.
    http://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile
    """
    
    f = 0.02/100.
    a = 0.5346
    b = 0.2166
    
    dT1 = np.power(a + np.multiply(np.multiply(b, dL)/np.sqrt(b*np.power(dL, 2) + np.power(dD, 2)), ddL), 2)
    dT2 = np.power(np.multiply(dD, ddD)/np.sqrt(b*np.power(dL, 2) + np.power(dD, 2)), 2)
    dT = np.sqrt(dT1 + dT2)
    
    dT = np.sqrt(np.power(dT, 2) + np.power(f*line_width(dD, dL), 2))
    
    return dT

def load_model(prop, specie, temp, dens, other=None):
    """
    Loads a model for the CRRL emission.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    ldir = "{0}/{1}/{2}".format(LOCALDIR, 'models', 'radtrans')
    if specie == 'alpha':
        if other:
            if 'width' in prop:
                mod_file = '{0}/Diffuse_CMB_bn_velturb_{1}_{2}_{3}_linewidth.dat'.format(ldir, temp, dens, other)
            else:
                mod_file = '{0}/Diffuse_CMB_bn_velturb_{1}_{2}_{3}_line.dat'.format(ldir, temp, dens, other)
        else:
            if 'width' in prop:
                mod_file = '{0}/Diffuse_CMB_bn_velturb_{1}_{2}_linewidth.dat'.format(ldir, temp, dens)
            else:
                mod_file = '{0}/Diffuse_CMB_bn_velturb_{1}_{2}_line.dat'.format(ldir, temp, dens)
    else:
        if other:
            if 'width' in prop:
                mod_file = '{0}/Diffuse_CMB_bn_velturb_{4}_{1}_{2}_{3}_linewidth.dat'.format(ldir, temp, dens, other, specie)
            else:
                mod_file = '{0}/Diffuse_CMB_bn_velturb_{4}_{1}_{2}_{3}_line.dat'.format(ldir, temp, dens, other, specie)
        else:
            if 'width' in prop:
                mod_file = '{0}/Diffuse_CMB_bn_velturb_{4}_{1}_{2}_linewidth.dat'.format(ldir, temp, dens, specie)
            else:
                mod_file = '{0}/Diffuse_CMB_bn_velturb_{4}_{1}_{2}_line.dat'.format(ldir, temp, dens, specie)
    
    model = np.loadtxt(mod_file)
    
    if 'width' in prop:
        
        qni = model[:,0]
        qnf = model[:,1]
        freq = model[:,2]
        dD = model[:,3]
        dLc = model[:,4]
        dLr = model[:,5]
        I = model[:,6]
        
        return np.array([qni, qnf, freq, dD, dLc, dLr, I])
    
    else:
        
        qn = model[:,0]
        freq = model[:,1]
        Ic = model[:,2]
        tau = model[:,3]
        eta_nu = model[:,4]
        
        return np.array([qn, freq, Ic, tau, eta_nu])

def load_ref(specie, trans):
    """
    Loads the reference spectrum for the
    specified atomic specie and transition.
    Available species and transitions: 
    CI alpha
    CI beta
    CI delta
    CI gamma
    CI13 alpha
    HeI alpha
    HeI beta
    HI alpha
    HI beta
    SI alpha
    SI beta
    """
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    refspec = np.loadtxt('{0}/linelist/RRL_{1}{2}.txt'.format(LOCALDIR, specie, trans),
                         usecols=(2,3))
                         #dtype={'formats': ('S4', 'S5', 'i4', 'f10')})
    qn = refspec[:,0]
    reffreq = refspec[:,1]
    
    return qn, reffreq

def load_ref2(transition):
    """
    Loads the reference spectrum for the
    specified atomic specie and transition.
    Available transitions: 
    CIalpha
    CIbeta
    CIdelta
    CIgamma
    CI13alpha
    HeIalpha
    HeIbeta
    HIalpha
    HIbeta
    SIalpha
    SIbeta
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    refspec = np.loadtxt('{0}/linelist/RRL_{1}.txt'.format(LOCALDIR, transition),
                         usecols=(2,3))
    qn = refspec[:,0]
    reffreq = refspec[:,1]
    
    return qn, reffreq

def lookup_freq(n, specie, trans):
    """
    Returns the frequency of a given transition.
    """
    
    qns, freqs = load_ref(specie, trans)
    indx = best_match_indx2(n, qns)
    
    return freqs[indx]

def lorentz_width(n, ne, Te, Tr, W, dn=1):
    """
    Gives the Lorentzian line width due to a combination
    of radiation and collisional broadening. The width
    is the FWHM in Hz. It uses the models of Salgado et al. (2015).
    """
    
    dL_r = radiation_broad_salgado(n, W, Tr)
    dL_p = pressure_broad_salgado(n, Te, ne, dn)
    
    return dL_r + dL_p

def mask_outliers(data, m=2):
    """
    Masks values larger than m times the data median.
    """
    return abs(data - np.median(data)) > m*np.std(data)

def natural_sort(l):
    """ 
    Sort the given list in the way that humans expect.
    Sorting is done in place.
    """
    l.sort(key=alphanum_key)

def n2f(n, line, n_min=1, n_max=1500, unitless=True):
    """
    Converts a given principal quantum number n to the 
    frequency of a given line.
    """
    
    line, nn, freq = make_line_list(line, n_min, n_max, unitless)
    nii = np.in1d(nn, n)
    
    return freq[nii]

def plot_spec_vel(out, x, y, fit, A, Aerr, x0, x0err, sx, sxerr):
    f = plt.figure(frameon=False)
    ax = f.add_subplot(1, 1, 1, adjustable='datalim')
    
    ax.plot(x, y, 'k-', drawstyle='steps', lw=1)
    ax.plot(x, fit, 'r--', lw=0.8)
    #ax.plot()
    
    ax.text(0.6, 0.1, 
            r"$\sigma v_{{line}}=${0:.1f}$\pm${1:.1f} km s$^{{-1}}$".format(sx, sxerr),
            size="large", transform=ax.transAxes, alpha=0.9)
    ax.text(0.6, 0.15, r"$v_{{line}}=${0:.1f}$\pm${1:.1f} km s$^{{-1}}$".format(x0, x0err), 
            size="large", transform=ax.transAxes, alpha=0.9)
    ax.text(0.6, 0.2, r"$\tau_{{peak}}=${0:.4f}$\pm${1:.4f}".format(A, Aerr), 
            size="large", transform=ax.transAxes, alpha=0.9)
    
    plt.xlabel(r"Radio velocity (km s$^{-1}$)")
    plt.ylabel(r"$\tau$")
        
    plt.savefig('{0}'.format(out), 
                bbox_inches='tight', pad_inches=0.3)
    plt.close()

def plot_model(x, y, xm, ym, out):
    
    f = plt.figure(frameon=False)
    ax = f.add_subplot(1, 1, 1, adjustable='datalim')
    
    ax.step(x, y, 'k-', drawstyle='steps', lw=1, where='pre')
    for i,j in zip(xm,ym):
        ax.step(i, j, '--', drawstyle='steps', lw=1, where='pre')
    
    ax.set_xlabel(r"Radio velocity (km s$^{-1}$)")
    ax.set_ylabel(r"$\tau$")
    
    plt.savefig('{0}'.format(out), 
                bbox_inches='tight', pad_inches=0.3)
    plt.close()

def plot_fit(fig, x, y, fit, params, vparams, sparams, rms, x0, \
             refs, refs_cb=None, refs_cd=None, refs_cg=None):
    
    fc = ['r', 'b', 'g']
    
    fig.suptitle('SB{0}, $n=${1:.0f}'.format(params['sb'], params['n']))
    
    ax = fig.add_subplot(2, 1, 1, adjustable='datalim')
    
    ax.step(x, y, 'k-', drawstyle='steps', lw=1, where='pre')
    #ax.plot(refs, [y.max()]*len(refs), 'kd', alpha=0.5)
    #if refs_cb:
    ax.plot(refs_cb, [y.max()]*len(refs_cb), 'rd', alpha=1, ls='none')
    ax.plot(refs_cd, [y.max()]*len(refs_cd), c='orange', marker='s', alpha=1, ls='none')
    ax.plot(refs_cd, [y.max()]*len(refs_cd), 'yd', ms=10, alpha=1, ls='none')
        
    for i,f in enumerate(fit):
        l = '{0}--'.format(fc[i])
        lr = '{0}:'.format(fc[i])
        ax.plot(x, f, l, lw=0.8)
        ax.plot(x, y-f, lr, lw=1)
    ax.plot([x[0],x[-1]], [params['tau0']]*2, 'k--')
    ax.plot([x[0],x[-1]], [params['tau0']-3*rms]*2, 'k:')
    #ax.plot()
    
    ax.text(0.6, 0.1, 
            r"FWHM$_{{line}}=${0:.1f}$\pm${1:.1f} km s$^{{-1}}$".format(params['dv1'], params['ddv1']),
            size="small", transform=ax.transAxes, alpha=0.9)
    ax.text(0.6, 0.17, r"$v_{{line}}=${0:.1f}$\pm${1:.1f} km s$^{{-1}}$".format(params['vp1'], params['dvp1']), 
            size="small", transform=ax.transAxes, alpha=0.9)
    ax.text(0.6, 0.23, r"$\tau_{{peak}}=${0:.4f}$\pm${1:.4f}".format(params['tau1'], params['dtau1']), 
            size="small", transform=ax.transAxes, alpha=0.9)
    ax.text(0.6, 0.33, r"$\int\tau dv=${0:.4f}$\pm${1:.4f}".format(params['itau1'], params['ditau1']), 
            size="small", transform=ax.transAxes, alpha=0.9)
    #ax.text(0.6, 0.25, r"$n=${0:.0f}".format(params['n']),
            #size="large", transform=ax.transAxes, alpha=0.9)
    #ax.text(0.6, 0.3, r"SB{0}".format(params['sb']),
            #size="large", transform=ax.transAxes, alpha=0.9)
    
    ax.set_xlabel(r"Radio velocity (km s$^{-1}$)")
    ax.set_ylabel(r"$\tau$")
    
    ax2 = fig.add_subplot(2, 1, 2, adjustable='datalim')
    
    ax2.plot(refs_cb, [y.max()]*len(refs_cb), 'rd', alpha=1, ls='none')
    ax2.plot(refs_cd, [y.max()]*len(refs_cd), c='orange', marker='s', alpha=1, ls='none')
    ax2.plot(refs_cd, [y.max()]*len(refs_cd), 'yd', ms=10, alpha=1, ls='none')
    
    ax2.step(x, y, 'k-', drawstyle='steps', lw=1, where='pre')
    for i,f in enumerate(fit):
        l = '{0}--'.format(fc[i])
        lr = '{0}:'.format(fc[i])
        ax2.plot(x, f, l, lw=0.8)
        ax2.plot(x, y-f, lr, lw=1)
    ax2.plot([x[0],x[-1]], [params['tau0']]*2, 'k--')
    ax2.plot([x[0],x[-1]], [params['tau0']-3*params['dtau0']]*2, 'k:')
    ch0 = sparams['ch0']
    chf = sparams['chf']
    ax2.fill_between(x[ch0:chf], y[ch0:chf], x0, facecolor='gray', alpha=0.5)
    
    vp = params['vp1']
    dv = abs(params['dv1'])
    #ax2.set_xlim(vp - 6*dv, vp + 6*dv)
    ax2.set_xlim(-400, 400)
    
    return fig

def plot_fit_single(fig, x, y, fit, params, rms, x0, \
             refs, refs_cb=None, refs_cd=None, refs_cg=None):
    
    fc = ['r', 'b', 'g']
    
    fig.suptitle('SB{0}, $n=${1:.0f}'.format(params['sb'], params['n']))
    
    ax = fig.add_subplot(2, 1, 1, adjustable='datalim')
    
    ax.step(x, y, 'k-', drawstyle='steps', lw=1, where='pre')
    #ax.plot(x, y, 'k-')
    #ax.plot(refs, [y.max()]*len(refs), 'kd', alpha=0.5)
    #if refs_cb:
    ax.plot(refs_cb, [y.max()]*len(refs_cb), 'rd', alpha=1, ls='none')
    ax.plot(refs_cd, [y.max()]*len(refs_cd), c='orange', marker='s', alpha=1, ls='none')
    ax.plot(refs_cd, [y.max()]*len(refs_cd), 'yd', ms=10, alpha=1, ls='none')
        
    for i,f in enumerate(fit):
        l = '{0}--'.format(fc[i])
        lr = '{0}:'.format(fc[i])
        ax.plot(x, f, l, lw=0.8)
        ax.plot(x, y-f, lr, lw=1)
    ax.plot([x[0],x[-1]], [params['tau0']]*2, 'k--')
    ax.plot([x[0],x[-1]], [params['tau0']-3*rms]*2, 'k:')
    #ax.plot()
    
    ax.text(0.6, 0.1, 
            r"FWHM$_{{line}}=${0:.1f}$\pm${1:.1f} km s$^{{-1}}$".format(params['dv1'], params['ddv1']),
            size="small", transform=ax.transAxes, alpha=0.9)
    ax.text(0.6, 0.17, r"$v_{{line}}=${0:.1f}$\pm${1:.1f} km s$^{{-1}}$".format(params['vp1'], params['dvp1']), 
            size="small", transform=ax.transAxes, alpha=0.9)
    ax.text(0.6, 0.23, r"$\tau_{{peak}}=${0:.6f}$\pm${1:.6f}".format(params['tau1'], params['dtau1']), 
            size="small", transform=ax.transAxes, alpha=0.9)
    ax.text(0.6, 0.33, r"$\int\tau dv=${0:.6f}$\pm${1:.6f}".format(params['itau1'], params['ditau1']), 
            size="small", transform=ax.transAxes, alpha=0.9)
    #ax.text(0.6, 0.25, r"$n=${0:.0f}".format(params['n']),
            #size="large", transform=ax.transAxes, alpha=0.9)
    #ax.text(0.6, 0.3, r"SB{0}".format(params['sb']),
            #size="large", transform=ax.transAxes, alpha=0.9)
    
    ax.minorticks_on()
    ax.set_xlabel(r"Radio velocity (km s$^{-1}$)")
    ax.set_ylabel(r"$\tau$")
    
    ax2 = fig.add_subplot(2, 1, 2, adjustable='datalim')
    
    ax2.plot(refs_cb, [y.max()]*len(refs_cb), 'rd', alpha=1, ls='none')
    ax2.plot(refs_cd, [y.max()]*len(refs_cd), c='orange', marker='s', alpha=1, ls='none')
    ax2.plot(refs_cd, [y.max()]*len(refs_cd), 'yd', ms=10, alpha=1, ls='none')
    
    ax2.step(x, y, 'k-', drawstyle='steps', lw=1, where='pre')
    for i,f in enumerate(fit):
        l = '{0}--'.format(fc[i])
        lr = '{0}:'.format(fc[i])
        ax2.plot(x, f, l, lw=0.8)
        ax2.plot(x, y-f, lr, lw=1)
    ax2.plot([x[0],x[-1]], [params['tau0']]*2, 'k--')
    ax2.plot([x[0],x[-1]], [params['tau0']-3*params['dtau0']]*2, 'k:')
    
    vp = params['vp1']
    dv = abs(params['dv1'])
    #ax2.set_xlim(vp - 6*dv, vp + 6*dv)
    ax2.set_xlim(-400, 400)
    ax2.minorticks_on()
    
    return fig

def pressure_broad(n, Te, ne):
    """
    Pressure induced broadening in Hz.
    Shaver (1975)
    """
    
    return 2e-5*np.power(Te, -3./2.)*np.exp(-26./np.power(Te, 1./3.))*ne*np.power(n, 5.2)

def pressure_broad_salgado(n, Te, ne, dn=1):
    """
    Pressure induced broadening in Hz.
    This gives the FWHM of a Lorentzian line.
    Salgado et al. (2015)
    """
    a, g = pressure_broad_coefs(Te)
    
    return ne*np.power(10., a)*(np.power(n, g) + np.power(n + dn, g))/2./np.pi

def pressure_broad_coefs(Te):
    
    te = [10, 20, 30, 40, 50, 60, 70, 80, 90,
          100, 200, 300, 400, 500, 600, 700,
          800, 900, 1000, 2000, 3000, 4000, 5000,
          6000, 7000, 8000, 9000, 10000, 20000, 30000]
    te_indx = best_match_indx2(Te, te)
    
    a = [-10.974098,           
         -10.669695,
         -10.494541,
         -10.370271,
         -10.273172,
         -10.191374,
         -10.124309,
         -10.064037,
         -10.010153,
         -9.9613006,
         -9.6200366,
         -9.4001678,
         -9.2336349,
         -9.0848840,
         -8.9690170,
         -8.8686695,
         -8.7802238,
         -8.7012421,
         -8.6299908,
         -8.2718376,
         -8.0093937,
         -7.8344941,
         -7.7083367,
         -7.6126791,
         -7.5375720,
         -7.4770500,
         -7.4272885,
         -7.3857095,
         -7.1811733,
         -7.1132522]
    
    gammac = [5.4821631,
              5.4354009,
              5.4071360,
              5.3861013,
              5.3689105,
              5.3535398,
              5.3409679,
              5.3290318,
              5.3180304,
              5.3077770,
              5.2283700,
              5.1700702,
              5.1224893,
              5.0770049,
              5.0408369,
              5.0086342,
              4.9796105,
              4.9532071,
              4.9290080,
              4.8063682,
              4.7057576,
              4.6356118,
              4.5831746,
              4.5421547,
              4.5090104,
              4.4815675,
              4.4584053,
              4.4385507,
              4.3290786,
              4.2814240]
    
    a_func = interpolate.interp1d(te, a,
                                  kind='linear',
                                  bounds_error=False,
                                  fill_value=0.0)
    
    g_func = interpolate.interp1d(te, gammac,
                                  kind='linear',
                                  bounds_error=False,
                                  fill_value=0.0)
    
    return [a_func(Te), g_func(Te)]
    
def radiation_broad(n, W, Tr):
    """
    Radiation induced broadening in Hz.
    """
    
    return 8e-17*W*Tr*np.power(n, 5.8)

def radiation_broad_salgado(n, W, Tr):
    """
    Radiation induced broadening in Hz.
    This gives the FWHM of a Lorentzian line.
    Salgado et al. (2015)
    """
    
    return 6.096e-17*W*Tr*np.power(n, 5.8)

def radiation_broad_salgado_general(n, W, Tr, nu0, alpha):
    """
    Radiation induced broadening in Hz.
    This gives the FWHM of a Lorentzian line.
    The expression is valid for power law like radiation fields.
    Salgado et al. (2015)
    """
    
    cte = 2./np.pi*2.14e4*np.power(6.578e15/nu0, alpha + 1.)*k_B.cgs.value*nu0
    dnexp = alpha - 2.
    
    return W*cte*Tr*np.power(n, -3*alpha - 2.)*(1 + np.power(2., dnexp) + np.power(3., dnexp))

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

def SavGol(y, **kwargs):
    #window_length, polyorder = args
    sgf = savgol_filter(y, window_length=kwargs['window_length'], 
                        polyorder=kwargs['polyorder'])
    return sgf

def sigma2FWHM(sigma):
    """
    Converts the sigma parameter of a Gaussian distribution
    to its FWHM.
    """
    return sigma*2*np.sqrt(2*np.log(2))

def sigma2FWHM_err(dsigma):
    """
    Converts the error on the sigma parameter of a Gaussian distribution
    to the error on the FWHM.
    """
    return dsigma*2*np.sqrt(2*np.log(2))

def stack_irregular(lines, window='', **kargs):
    """
    Stacks spectra by adding them together and 
    then convolving with a window to reduce 
    the noise.
    Available window functions:
    Gaussian, Savitzky-Golay and Wiener.
    """
    
    vgrid = []
    tgrid = []
    ngrid = []
    
    # Loop over the spectra to stack
    for line in lines:
        data = np.loadtxt(line)
        vel = data[:,0] # Velocity in km/s
        tau = data[:,1] # Intensity in optical depth units
        
        # Catch NaNs and invalid values:
        mask_v = np.ma.masked_equal(vel, 1.0).mask
        mask_t = np.isnan(tau)
        mask = np.array(reduce(np.logical_or, [mask_v, mask_t]))
        
        # Remove NaNs and invalid values
        vel = vel[~mask]
        tau = tau[~mask]
        
        # Sort by velocity
        vel, tau = (list(t) for t in zip(*sorted(zip(vel, tau))))
        
        # Append
        vgrid.append(vel)
        tgrid.append(tau)
        
    # Flatten the stacked spectra
    vgrid = list(itertools.chain.from_iterable(vgrid))
    tgrid = list(itertools.chain.from_iterable(tgrid))
    
     # Sort by velocity
    vgrid, tgrid = (list(t) for t in zip(*sorted(zip(vgrid, tgrid))))
    
    # Apply a window function to reduce noise
    if window == '':
        print 'Using default Gaussian window.'
        stau = Gauss(tgrid, {'sigma':3, 'order':0})
    elif window == 'Wiener':
        stau = Wiener(tgrid, **kargs)
    elif window == 'SavGol':
        stau = SavGol(tgrid, **kargs)
    elif window == 'Gauss':
        stau = Gauss(tgrid, **kargs)
 
    return vgrid, tgrid, stau
        
def stack_interpol(spectra, vmin, vmax, dv, show=True, rmsvec=False):
    
    vgrid = np.arange(vmin, vmax, dv)
    tgrid = np.zeros(len(vgrid))      # the temperatures
    ngrid = np.zeros(len(vgrid))      # the number of tb points in every stacked channel
    crms = np.zeros(len(spectra))
    snr = np.zeros(len(spectra))
    nspec = np.arange(len(spectra)) + 1 
    nlist = np.arange(len(spectra))
    
    for s,spec in enumerate(spectra):
        if show:
            print "Working on file: {0}".format(spec)
        nlist[s] = int(re.findall('\d+', spec.split('/')[-1])[0])
        data = np.loadtxt(spec)
        vel = data[:,1] # velocity in km/s
        tau = data[:,2]  # optical depth
        rms = data[0,3] # continuum rms
        
        # Sort by velocity
        vel, tau = (list(t) for t in zip(*sorted(zip(vel, tau))))
        vel = np.array(vel)
        tau = np.array(tau)
        
        # Catch NaNs and invalid values:
        mask_v = np.ma.masked_equal(vel, 1.0).mask
        mask_tb = np.isnan(tau)
        mask = np.array(reduce(np.logical_or, [mask_v, mask_tb]))
        
        # Interpolate non masked ranges indepently
        mtau = np.ma.masked_where(mask, tau)
        mvel = np.ma.masked_where(mask, vel)
        valid = np.ma.flatnotmasked_contiguous(mvel)
        itb = np.zeros(len(vgrid))
        if not isinstance(valid, slice):
            for i,rng in enumerate(valid):
                #print "slice {0}: {1}".format(i, rng)
                if len(vel[rng]) > 1:
                    interp_tb = interpolate.interp1d(vel[rng], tau[rng],
                                                     kind='linear',
                                                     bounds_error=False,
                                                     fill_value=0.0)
                    itb += interp_tb(vgrid)
                else:
                    itb[best_match_indx(vel[rng], vgrid, dv/2.0)] += tau[rng]
        else:
            #print "slice: {0}".format(valid)
            interp_tb = interpolate.interp1d(vel[valid], tau[valid],
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value=0.0)
            itb += interp_tb(vgrid)
        
        # Check the velocity coverage
        chstack = [1 if ch != 0 else 0 for ch in itb]
        
        # Stack!
        w = [1/rms]*len(ngrid)
        #print w
        ngrid = ngrid + np.multiply(chstack, w)
        tgrid = tgrid + itb*np.multiply(chstack, w)
        
        sspec = np.divide(tgrid, ngrid)
        svel = vgrid[~np.isnan(sspec)]
        sspec = sspec[~np.isnan(sspec)]
        vii = best_match_indx2(vmin, svel)
        vfi = best_match_indx2(-200, svel)
        rmsl = get_rms(sspec[vii+1:vfi])
        vii = best_match_indx2(150, svel)
        vfi = best_match_indx2(vmax, svel)
        rmsr = get_rms(sspec[vii:vfi-1])
        crms[s] = min(rmsl, rmsr)
        snr[s] = -min(sspec[5:-5])/crms[s]
            
    # Divide by ngrid to preserve optical depth
    tgrid = np.divide(tgrid, ngrid)
    
    # Compute the variation of the rms as we stack
    if len(crms) > 2:
        mod = PowerLawModel()
        parms = mod.make_params(amplitude=crms[0], exponent=-0.5)
        fit = mod.fit(crms, parms, x=nspec)
        best_fit = fit.best_fit
        if show:
            #print nlist
            print fit.fit_report()
            print "Lowest rms: {0} for # stacks: {1}".format(min(crms), len(crms))
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            #ax2 = ax1.twiny()
            ax1.plot(nlist[::1], crms, 'bo')
            ax1.plot(nlist[::1], best_fit, 'r--', label='best fit')
            #ax1.plot(nlist[::1], mod.eval(x=nspec, amplitude=fit.params['amplitude'].value, exponent=-0.5), 
                     #'r:', label=r'$1/\sqrt{\#}$')
            ax1.plot(nlist[::1], mod.eval(x=nspec, amplitude=crms[0], exponent=-0.5), 
                     'r:', label=r'$1/\sqrt{\#}$')
            ax1.set_xlabel(r'# stacks')
            ax1.set_ylabel(r'Continuum rms')
            #ax2.plot(nlist, np.ones(len(nlist))*min(crms), alpha=0)
            ax1.legend(loc=0, numpoints=1, frameon=False)
            #ax2.cla()
            plt.show()
            
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(nlist[::1], snr, 'bo')
            ax1.set_xlabel(r'# stacks')
            ax1.set_ylabel(r'SNR')
            plt.show()
    
    if rmsvec:
        return vgrid[1:-1], tgrid[1:-1], ngrid[1:-1], nlist[::1], crms
    else:
        return vgrid[1:-1], tgrid[1:-1], ngrid[1:-1]

def sum_line(sb, n, ref, vel, tau, v0, tau0, dtau0, thr, rms):
    """
    Integrate the spectrum near a given velocity v0.
    It stops when the channels are within a threshold
    from a reference level.
    """
    
    results = sum_storage()
    
    results['n'] = n
    results['sb'] = sb
    results['reffreq'] = ref
    results['rms'] = rms
    
    # The integrated optical depth starts at a value of 0
    itau = 0
    
    # Find where we start counting channels
    dv = min(abs(vel[0:-1:2] - vel[1::2]))
    v0_indx = best_match_indx2(v0, vel)
    #print "v0_indx: {0}".format(v0_indx)
    
    # Start adding to negative velocities
    ch0 = 0
    for i in range(v0_indx):
        if not (v0_indx-i < 0):
            if tau[v0_indx-i] <= tau0 - thr*dtau0:
                itau += tau[v0_indx-i]*dv
            else:
                ch0 = v0_indx-i
                break
        else:
            ch0 = 0
            break
    # Now to positive velocities
    chf = 0
    for i in range(len(vel)):
        if not (v0_indx+i > len(vel) - 1):
            if tau[v0_indx+i] <= tau0 - thr*dtau0:
                itau += tau[v0_indx+i]*dv
            else:
                chf = v0_indx+i
                break
        else:
            chf = len(vel) - 1
            break
        
    print "ch0: {0} chf: {1}".format(ch0, chf)
    
    if ch0 != chf:
        results['tau'] = min(tau[ch0:chf])
        results['vp'] = vel[np.where(tau == results['tau'])[0]][0]
    results['dtau'] = dtau0
    
    #print "Sum vp: {0}".format(results['vp'])
    results['dvp'] = dv
    results['dv'] = abs(vel[chf] - vel[ch0])
    results['ddv'] = dv # Bad assumption
    results['itau'] = itau
    results['ditau'] = abs(ch0 - chf)*rms
    results['ch0'] = ch0
    results['chf'] = chf
    results['tau0'] = tau0 - thr*dtau0
        
    return results

def sum_storage():
    blankval = -99
    results = collections.OrderedDict((('sb',blankval),      #0 Sub Band number
                                       ('n',blankval),       #1 Principal quantum number
                                       ('reffreq',blankval), #2 Reference frequency
                                       ('tau',blankval),     #3 Peak optical depth
                                       ('vp',blankval),      #4 Velocity of peak optical depth
                                       ('dv',blankval),      #5 Line width
                                       ('itau',blankval),    #6 Integrated optical depth
                                       ('tau0',blankval),    #7 Optical depth cutoff
                                       ('dtau',blankval),    #8 Peak optical depth error
                                       ('dvp',blankval),     #9 Velocity of peak optical depth error
                                       ('ddv',blankval),     #10 Line width error
                                       ('ditau',blankval),   #11 Integrated optical depth error
                                       ('dtau0',blankval),   #12 Optical depth offset error
                                       ('ch0',blankval),     #13
                                       ('chf',blankval),     #14
                                       ('rms',blankval)))    #15 Continuum rms
                                       
    return results

def tryint(s):
    try:
        return int(s)
    except:
        return s

def vel2freq(f0, vel):
    """
    Convert a velocity axis
    to a frequency axis given
    a central frequency.
    Uses the radio definition.
    """
    return f0*(1. - vel/c)

def voigt(x, y):
    # The Voigt function is also the real part of 
    # w(z) = exp(-z^2) erfc(iz), the complex probability function,
    # which is also known as the Faddeeva function. Scipy has 
    # implemented this function under the name wofz()

    z = x + 1j*y
    I = wofz(z).real

    return I

def Voigt(x, sigma, gamma, center, amplitude):
    """
    The Voigt line shape in terms of its physical parameters
    x: independent variable
    sigma: HWHM of the Gaussian
    gamma: HWHM of the Lorentzian
    center: the line center
    amplitude: the line area
    """

    ln2 = np.log(2)
    f = np.sqrt(ln2)
    rx = (x - center)/sigma * f
    ry = gamma/sigma * f

    V = amplitude*f/(sigma*np.sqrt(np.pi)) * voigt(rx, ry)

    return V

def voigt_area(amp, fwhm, gamma, sigma):
    """
    Returns the area under a Voigt profile.
    This approximation has an error of less than 0.5%
    """
    
    l = 0.5*gamma
    g = np.sqrt(2*np.log(2))*sigma
    k = g/(g+l)
    c = 1.572 + 0.05288*k + -1.323*k**2 + 0.7658*k**3
    
    return c*amp*fwhm

def voigt_area_err(area, amp, damp, fwhm, dfwhm, gamma, sigma):
    """
    Returns the error of the area under a Voigt profile.
    Assumes that the parameter c has an error of 0.5%.
    """
    
    l = 0.5*gamma
    g = np.sqrt(2*np.log(2))*sigma
    k = g/(g+l)
    c = 1.572 + 0.05288*k + -1.323*k**2 + 0.7658*k**3
    
    err_a = area/amp*damp
    err_f = area/fwhm*dfwhm
    err_c = area/c*0.5/100.0
    
    err = np.sqrt(err_a**2 + err_f**2 + err_c**2)
    
    return err

def voigt_peak(A, alphaD, alphaL):
    """
    Gives the peak of a Voigt profile given
    its Area and the HWHM of the Gaussian and
    Lorentz profiles.
    """
    
    y = alphaL/alphaD*np.sqrt(np.log(2))
    z = 0 + 1j*y
    K = wofz(z).real
    
    peak = A/alphaD*np.sqrt(np.log(2)/np.pi)*K
    
    return peak

def voigt_peak2area(peak, alphaD, alphaL):
    """
    Converts the peak of a Voigt profile into the area under the profile
    given the HWHM of the profile.
    """
    
    y = alphaL/alphaD*np.sqrt(np.log(2))
    z = 0 + 1j*y
    K = wofz(z).real
    
    A = peak*alphaD/(np.sqrt(np.log(2)/np.pi)*K)
    
    return A
    

def voigt_peak_err(peak, A, dA, alphaD, dalphaD):
    """
    Gives the error on the peak of
    the Voigt profile.
    """
    
    dpeak = abs(peak)*np.sqrt(np.power(dalphaD/alphaD ,2) + np.power(dA/A ,2))
    
    return dpeak

def Wiener(y, **kwargs):
    #size, noise = args
    wi = wiener(y, mysize=kwargs['mysize'], noise=kwargs['noise'])
    return wi
