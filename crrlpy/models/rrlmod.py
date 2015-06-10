#!/usr/bin/env python

import numpy as np
import os
import glob
import re
from crrlpy import frec_calc as fc
from astropy.constants import h, k_B
from decimal import *
from crrlpy.crrls import natural_sort
getcontext().prec = 450

def itau_all(trans='alpha', n_max=1000):
    """
    Loads all the available models.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    models = glob.glob('{0}/bbn/*'.format(LOCALDIR))
    natural_sort(models)
    
    Te = np.zeros(len(models))
    ne = np.zeros(len(models))
    other = np.zeros(len(models), dtype='|S20')
    data = np.zeros((len(models),2,n_max-3))
    for i,model in enumerate(models):
        st = model[model.index('T')+2:model.index('T')+5]
        Te[i] = str2val(st)
        sn = model[model.index('ne')+3:model.index('ne')+7].split('_')[0]
        ne[i] = str2val(sn)
        other[i] = model.split('bn_beta')[-1]
        #mod = np.loadtxt(model)
        #mod = mod[:np.where(mod[:,0]==n_max)[0]]
        n, int_tau = itau(st, sn, trans, n_max=1000, other=other[i])
        #data[i,0] = mod[:,0]
        #data[i,1] = mod[:,1]
        data[i,0] = n
        data[i,1] = int_tau
        
    return [Te, ne, other, data]

def itau_all_norad(trans='alpha', n_max=1000):
    """
    Loads all the available models.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    models = glob.glob('{0}/bbn/*_dat_bn_beta'.format(LOCALDIR))
    natural_sort(models)
    
    Te = np.zeros(len(models))
    ne = np.zeros(len(models))
    other = np.zeros(len(models), dtype='|S20')
    data = np.zeros((len(models),2,n_max-3))
    for i,model in enumerate(models):
        st = model[model.index('T')+2:model.index('T')+5]
        Te[i] = str2val(st)
        sn = model[model.index('ne')+3:model.index('ne')+7].split('_')[0]
        ne[i] = str2val(sn)
        other[i] = model.split('bn_beta')[-1]
        #mod = np.loadtxt(model)
        #mod = mod[:np.where(mod[:,0]==n_max)[0]]
        n, int_tau = itau(st, sn, trans, n_max=1000, other=other[i])
        #data[i,0] = mod[:,0]
        #data[i,1] = mod[:,1]
        data[i,0] = n
        data[i,1] = int_tau
        
    return [Te, ne, other, data]

def itau(temp, dens, trans, n_max=1000, other=''):
    """
    Gives the integrated optical depth for a given temperature and density. 
    The emission measure is unity. The output units are Hz.
    """
    
    bbn = load_betabn(temp, dens, other)
    n = bbn[:,0]
    b = bbn[:,1]
    
    b = b[:np.where(n==n_max)[0]]
    
    
    t = str2val(temp)
    d = str2val(dens)
    
    dn = fc.set_dn(trans)
    mdn = Mdn(dn)
    
    # Convert the betabn values to the corresponding transition
    if 'alpha' not in trans:
        
        #specie, trans, n, freq = fc.make_line_list('CI', n_max, dn)
        #bn = load_bn(temp, dens, other='')
        #beta = (1 - np.divide(bn[dn::], bn[0::])*np.exp(-h*freq*1e6/(k_B*t)))/ \
               #(1 - np.exp(-h*freq*1e6/(k_B*t)))
        b = make_betabn(temp, dens, trans, n_max=n_max+1, other='')[1]
    n = n[:np.where(n==n_max)[0]]
    
    i = -1.069e7*dn*mdn*b*np.exp(1.58e5/(np.power(n, 2)*t))/np.power(t, 5./2.)
    
    return n, i

def load_betabn(temp, dens, other=''):
    """
    Loads a model for the CRRL emission.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    mod_file = '{0}/bbn/Carbon_opt_T_{1}_ne_{2}_ncrit_1.5d3_vriens_delta_500_vrinc_nmax_9900_dat_bn_beta{3}'.format(LOCALDIR, temp, dens, other)
    
    data = np.loadtxt(mod_file)
    
    return data

def load_bn(temp, dens, other=''):
    """
    Loads the bn values from the CRRL models.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    mod_file = '{0}/bn/Carbon_opt_T_{1}_ne_{2}_ncrit_1.5d3_vriens_delta_500_vrinc_nmax_9900_dat{3}'.format(LOCALDIR, temp, dens, other)
    
    data = np.loadtxt(mod_file)
    
    return data

def make_betabn(temp, dens, trans, n_max=1000, other=''):
    """
    """
    
    t = str2val(temp)
    d = str2val(dens)
    
    dn = fc.set_dn(trans)
    bn = load_bn(temp, dens, other='')
    specie, trans, n, freq = fc.make_line_list('CI', bn[-1,0]+1, dn)
    # Cut bn first
    bn = bn[:np.where(bn[:,0]==n_max)[0]]
    freq = freq[np.where(n==bn[0,0])[0]:np.where(n==bn[-1,0])[0]]
    
    beta = np.empty(len(freq))
    for i in xrange(len(freq)):
        if i < len(freq)-dn:
            #bnn = np.divide(bn[i+dn,-1], bn[i,-1])
            bnn = Decimal(bn[i+dn,-1]) / Decimal(bn[i,-1])
            e = -h.value*freq[i]*1e6/(k_B.value*t)
            exp = Decimal(e).exp()#Decimal(np.exp(-h.value*freq[i]*1e6/(k_B.value*t)))
            beta[i] = float((Decimal(1) - bnn*exp)/(Decimal(1) - exp))
        
    return np.array([bn[:-1,0], beta*bn[:-1,1]])
    
def Mdn(dn):
    """
    Gives the M(dn) factor for a given dn.
    ref. Menzel (1968)
    """
    
    if dn == 1:
        mdn = 0.1908
    if dn == 2:
        mdn = 0.02633
    if dn == 3:
        mdn = 0.008106
    if dn == 4:
        mdn = 0.003492
    if dn == 5:
        mdn = 0.001812
        
    return mdn

def str2val(str):
    """
    """
    
    aux = map(float, str.split('d'))
    val = aux[0]*np.power(10., aux[1])
    
    return val

def val2str(val):
    """
    Converts a float to the str format required for loading the
    CRRL models. E.g., a temperature of 70 K is 7d1.
    """
    
    d = np.floor(np.log10(val))
    u = val/np.power(10., d)
    
    if u.is_integer():
        return "{0:.0f}d{1:.0f}".format(u, d)
    else:
        return "{0}d{1:.0f}".format(u, d)