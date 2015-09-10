#!/usr/bin/env python

import numpy as np
import argparse
from crrlpy import crrls
from crrlpy.models import rrlmod
from crrlpy import synthspec as synth
import pylab as plt

def make_spec(spec, fi, bw, n, rms, v0, transitions, Te, ne, Tr, W, dD, EM, n_max=1500, verbose=False):
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
    baseline = synth.make_baseline(n*2, n/250.)
    tau_b = tau_n
    
    z = v0/3e5
    for i,trans in enumerate(transitions.split(',')):
        
        n_l, f_l = crrls.find_lines_sb(freq, trans, z)
        
        # Generate the line properties as a function of n
        dL_r = crrls.radiation_broad_salgado(n_l, W, Tr)/1e6
        dL_p = crrls.pressure_broad_salgado(n_l, Te, ne)/1e6
        dL = dL_r + dL_p
        n_itau, a_itau = rrlmod.itau(rrlmod.val2str(Te), 
                                     ne, 
                                     trans[2:], n_max=n_max, 
                                     other='case_diffuse_{0}'.format(rrlmod.val2str(Tr)),
                                     verbose=verbose)
    
        dD_f = crrls.dv2df(f_l, dD*1e3)
        
        for j,f in enumerate(f_l):
            itau = a_itau[np.where(n_itau==n_l[j])[0]][0]/1e6
            line = crrls.Voigt(freq, dD_f[j]/2., dL[j]/2., f, itau*EM)
            if verbose:
                print "Line properties:"
                print("f: {0}, A: {1}, dD: {2}, dD_f/2: {3}, " \
                    "dL/2: {4}".format(f, itau*EM, dD, dD_f[j]/2., dL[j]/2.))
            tau_b += line
            
    np.savetxt(spec, np.c_[freq, tau_b])
    
    
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
    parser.add_argument('-w', type=float,
                        help="Cloud covering factor. (float)")
    parser.add_argument('-e', '--EM', type=float,
                        help="Emission meassure in pc cm^-6. (float)")
    parser.add_argument('-d', '--doppler', type=float,
                        help="Doppler FWHM in km/s. (float)")
    parser.add_argument('-v', '--v0', type=float, default=0,
                        help="Cloud velocity in km/s. (float)")
    parser.add_argument('-r', '--rms', type=float,
                        help="Spectral rms in optical depth units. (float)")
    parser.add_argument('--transitions', type=str,
                        help="Spectral rms in optical depth units. (string)")
    args = parser.parse_args()
    
    make_spec(args.spec, args.fi, args.bw, args.nchan, args.rms, args.v0, 
              args.transitions, args.Te, args.ne, args.Tr, args.w, args.doppler, args.EM)
