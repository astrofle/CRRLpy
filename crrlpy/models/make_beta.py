#!/usr/bin/env python

from crrlpy import crrls
from crrlpy.models import rrlmod
from crrlpy import frec_calc as fc
from astropy.constants import h, k_B
from mpmath import mp
mp.dps = 50 

import glob
import os

import numpy as np

if __name__ == '__main__':
    
    cwd = os.getcwd()
    os.chdir('bn2')
    
    line = 'CIgamma'
    files = glob.glob('*')
    h = mp.mpf(h.value)
    k_B = mp.mpf(k_B.value)
    
    for f in files:
        
        t = mp.mpf(rrlmod.str2val(f.split('_')[3]))
        
        data =  np.loadtxt(f, dtype=str)
        bn = data[:,-1]
        freq = crrls.n2f(map(float, data[:,0]), line, n_max=float(data[-1,0])+1)
        dn = fc.set_dn(line)
        
        beta = np.empty(len(bn), dtype=mp.mpf)
        betabn = np.empty(len(bn), dtype=mp.mpf)
        for i in xrange(len(bn)):
            if i < len(bn)-dn:
                #bnn = np.divide(bn[i+dn,-1], bn[i,-1])
                nu = mp.mpf(freq[i])
                bnn = mp.mpf(bn[i+dn]) / mp.mpf(bn[i])
                e = mp.mpf(-h*nu*1e6/(k_B*t))
                exp = mp.exp(e)
                beta[i] = (mp.mpf('1') - bnn*exp)/(mp.mpf('1') - exp)
                if beta[i]*mp.mpf(bn[i]) != 'None':
                    betabn[i] = beta[i]*mp.mpf(bn[i])
                else:
                    betabn[i] = -9999
                
        np.savetxt('{0}/bbn2_{1}/{2}bn_beta'.format(cwd, line, f), np.column_stack((data[:-30,0], betabn[:-30])), fmt=('%s', '%s'))
        
    os.chdir(cwd)
