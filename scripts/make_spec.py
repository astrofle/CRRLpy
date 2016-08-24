#!/usr/bin/env python


import argparse
import os
import matplotlib as mpl
havedisplay = "DISPLAY" in os.environ
if not havedisplay:
    mpl.use('Agg')

import numpy as np
import pylab as plt

from crrlpy import crrls
from crrlpy.models import rrlmod
from crrlpy import synthspec as synth

def power_law(freq, Tc, nu):
    """
    """
    
    return Tc*np.power(freq/100., nu)

def make_spec(fi, bw, n, rms, v0, transitions, bandpass, baseline, order,
              Te, ne, Tr, W, dD, EM, cont, Tc, nu, plot, plot_out,
              n_max=1500, verbose=False):
    """
    Generates a synthetic spectrum given an initial frequency a bandwidth and number of channels.
    The synthetic spectrum will have Gaussian white noise across it and a nonlinear baseline.
    The spectrum will be populated with RRL given the models of Salgado et al. (2015).
    The line velocity can be specified as well as the transitions. The line model requires an
    electron temperature, density, a temperature for the external radiation field, a covering factor
    and an emission meassure.
    For computing the line width of the profile a Doppler factor has to be specified.
    """
    
    ff = fi + bw
    df = bw/n
    
    # Make the arrays to store the spectrum
    freq = np.arange(fi, ff, df)
    tau = np.zeros(n)
    
    # Create Gaussian noise and add it to the spectrum.
    noise = synth.make_noise(0, rms, n)
    tau_n = tau + noise
    
    # Create a baseline that mimics a standing wave 
    # and add it to the spectrum.
    # This values produce a nice variation.
    if bandpass:
        tau_n += synth.make_ripples(4*n, n/2., n, rms)
    
    if baseline:
        tau_n += synth.make_offset(rms, freq, order=order)
    
    z = v0/3e5
    for i,trans in enumerate(transitions.split(',')):
        if trans != '':
            n_l, f_l = crrls.find_lines_sb(freq, trans, z, verbose)
            
            # Generate the line properties as a function of n
            dL_r = crrls.radiation_broad_salgado(n_l, W, Tr)/1e6
            dL_p = crrls.pressure_broad_salgado(n_l, Te, ne)/1e6
            dL = dL_r + dL_p
            
            if Tr != 0:
                other = 'case_diffuse_{0}'.format(rrlmod.val2str(Tr))
            else:
                other = ''
            
            n_itau, a_itau = rrlmod.itau(rrlmod.val2str(Te), 
                                        ne, 
                                        trans, n_max=n_max, 
                                        other=other,
                                        verbose=verbose)
        
            dD_f = crrls.dv2df(f_l, dD*1e3)
            
            for j,f in enumerate(f_l):
                itau = a_itau[np.where(n_itau==n_l[j])[0]][0]/1e6
                line = crrls.voigt(freq, dD_f[j]/2., dL[j]/2., f, itau*EM)
                if verbose:
                    print "Line properties:"
                    print("f: {0}, A: {1}, dD: {2}, dD_f/2: {3}, " \
                        "dL/2: {4}".format(f, itau*EM, dD, dD_f[j]/2., dL[j]/2.))
                tau_n += line
            
    if cont:
        tau_n = (1 + tau_n)*power_law(freq, Tc, nu)
        
    if plot:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(freq, tau_n, 'k-', label='generated spectrum')
        ax.plot(freq, power_law(freq, Tc, nu), 'r-', label='continuum')
        plt.legend(loc=0, numpoints=1, frameon=False)
        plt.savefig('{0}'.format(plot_out), 
                bbox_inches='tight', pad_inches=0.3)
        plt.close()
        
        
    #np.savetxt(spec, np.c_[freq, tau_n])
    
    return freq, tau_n
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('spec', type=str,
                        help="Spectrum name. (string)")
    parser.add_argument('-f', '--fi', type=float,
                        help="Starting frequency in MHz. (float)")
    parser.add_argument('-b', '--bw', type=float,
                        help="Bandwidth in MHz. (float)")
    parser.add_argument('-n', '--nchan', type=int,
                        help="Number of channels. (int)")
    parser.add_argument('-t', '--Te', type=float,
                        help="Electron temperature. (float)")
    parser.add_argument('--ne', type=float,
                        help="Electron density. (float)")
    parser.add_argument('--Tr', type=float,
                        help="Radiation field in K. (float)")
    parser.add_argument('-w', type=float, default=1,
                        help="Cloud covering factor. (float)\n" \
                             "Default: 1")
    parser.add_argument('-e', '--EM', type=float,
                        help="Emission meassure in pc cm^-6. (float)")
    parser.add_argument('-d', '--doppler', type=float, default=2.,
                        help="Doppler FWHM in km/s. (float)\n" \
                             "Default: 2")
    parser.add_argument('-v', '--v0', type=float, default=0,
                        help="Cloud velocity in km/s. (float)\n" \
                             "Default: 0")
    parser.add_argument('-r', '--rms', type=float,
                        help="Spectral rms in optical depth units. (float)")
    parser.add_argument('--transitions', type=str, default='CIalpha',
                        help="Transitions to include in the simulated spectrum.\n" \
                             "E.g. 'CIalpha,CIbeta,HIalpha'. (string)\n" \
                             "Default: CIalpha")
    parser.add_argument('--bandpass', action='store_true',
                        help="Add a bandpass filter response to the spectrum?")
    parser.add_argument('--baseline', action='store_true',
                        help="Add a baseline offset to the spectrum?")
    parser.add_argument('--order', type=int, default=None,
                        help="Baseline order to add to the spectrum.")
    parser.add_argument('--cont', action='store_true',
                        help="Add continuum to the spectra?.\n" \
                             "Assumes a power law spectrum with Tc at 100 MHz.")
    parser.add_argument('--Tc', type=float,
                        help="Power law continuum intensity at 100 MHz. (float)")
    parser.add_argument('--plawexp', type=float, default=-2.6,
                        help="Continuum power law exponent.")
    parser.add_argument('--plot', action='store_true',
                        help="Plot the generated spectrum?")
    parser.add_argument('--plot_out', type=str,
                        help="Plot name. (string)")
    args = parser.parse_args()
    
    make_spec(args.spec, args.fi, args.bw, args.nchan, args.rms, args.v0, 
              args.transitions, args.baseline, args.order,
              args.Te, args.ne, args.Tr, args.w, args.doppler, args.EM, 
              args.cont, args.Tc, args.plawexp, args.plot, args.plot_out)
