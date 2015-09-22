#!/usr/bin/env python

from __future__ import division

from crrlpy import crrls
from crrlpy.models import rrlmod
from scripts import make_spec
import numpy as np
import sys

def rms_model(nu, nu_rms, rms):
    """
    """

    return (1/3)*rms[crrls.best_match_indx2(nu, nu_rms)]
    
if __name__ == '__main__':
    
    rmsmod, nu_rms = np.loadtxt('lba_rms.log', usecols=[1,2], unpack=True)

    dest = sys.argv[1]
    
    f0 = 10
    bw = 1.953125e-1
    n = 512
    v0a = -47.0
    v0b = -38.0
    v0c = 0.0
    Te = float(sys.argv[2])
    Tr = float(sys.argv[3])
    ne = float(sys.argv[4])
    W = 1
    dD = 3.4
    EMa = 7.5e-2
    EMb = 4e-2
    EMc = 1e-2
    transitions='CIalpha,CIbeta,CIgamma,CIdelta'
    bandpass = True
    baseline = True
    order = 1
    cont = True
    Tc = 400
    nu_cont = -2.6
    plot = False
 
    for i in xrange(0, 360):
        
        fi = f0 + bw*i
        specnl = '{0}lba_sim_SB{1:03d}nl.ascii'.format(dest, i)
        speca = '{0}lba_sim_SB{1:03d}a.ascii'.format(dest, i)
        specb = '{0}lba_sim_SB{1:03d}b.ascii'.format(dest, i)
        specc = '{0}lba_sim_SB{1:03d}c.ascii'.format(dest, i)
        spec = '{0}lba_sim_SB{1:03d}.ascii'.format(dest, i)

        #print spec, fi
        rms = rms_model(fi, nu_rms, rmsmod)
        make_spec.make_spec(specnl, fi, bw, n, 1e-10, 0, '', bandpass, baseline, order, Te, ne, Tr, W, dD, EMa, cont, Tc, nu_cont, plot, '{0}nl.pdf'.format(i), verbose=False)
        make_spec.make_spec(speca, fi, bw, n, rms, v0a, transitions, bandpass, baseline, order, Te, ne, Tr, W, dD, EMa, False, 0, nu_cont, plot, '{0}a.pdf'.format(i), verbose=False)
        make_spec.make_spec(specb, fi, bw, n, rms, v0b, transitions, bandpass, baseline, order, Te, ne, Tr, W, dD, EMb, False, 0, nu_cont, plot, '{0}b.pdf'.format(i), verbose=False)
        make_spec.make_spec(specc, fi, bw, n, rms, v0c, transitions, bandpass, baseline, order, Te, ne, Tr, W, dD, EMc, False, 0, nu_cont, plot, '{0}c.pdf'.format(i), verbose=False)

        data_nl = np.loadtxt(specnl)
        data_a = np.loadtxt(speca)
        data_b = np.loadtxt(specb)
        data_c = np.loadtxt(specc)

        data = (data_a[:,1] + data_b[:,1] + data_c[:,1] + 1.)*data_nl[:,1]

        np.savetxt(spec, np.c_[data_a[:,0], data])

      
 
