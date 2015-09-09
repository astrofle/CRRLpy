#!/usr/bin/env python

import numpy as np
import os
import glob
import re
import filecmp
from crrlpy import frec_calc as fc
from astropy.constants import h, k_B
from decimal import *
from crrlpy.crrls import natural_sort
getcontext().prec = 450

LOCALDIR = os.path.dirname(os.path.realpath(__file__))



#def itau(temp, dens, trans, n_max=1000, other=''):
    #"""
    #Gives the integrated optical depth for a given temperature and density. 
    #The emission measure is unity. The output units are Hz.
    
    #:returns: principal quantum number and integrated optical depth.
    #:rtype: numpy arrays.
    #"""
    
    #bbn = load_betabn(temp, dens, other)
    #n = bbn[:,0]
    #b = bbn[:,1]
    
    #b = b[:np.where(n==n_max)[0]]
    
    
    #t = str2val(temp)
    #d = str2val(dens)
    
    #dn = fc.set_dn(trans)
    #mdn = Mdn(dn)
    
    ## Convert the betabn values to the corresponding transition
    #if 'alpha' not in trans:
        
        ##specie, trans, n, freq = fc.make_line_list('CI', n_max, dn)
        ##bn = load_bn(temp, dens, other='')
        ##beta = (1 - np.divide(bn[dn::], bn[0::])*np.exp(-h*freq*1e6/(k_B*t)))/ \
               ##(1 - np.exp(-h*freq*1e6/(k_B*t)))
        #b = make_betabn(temp, dens, trans, n_max=n_max+1, other='')[1]
    #n = n[:np.where(n==n_max)[0]]
    
    #i = -1.069e7*dn*mdn*b*np.exp(1.58e5/(np.power(n, 2)*t))/np.power(t, 5./2.)
    
    #return n, i

def itau(temp, dens, trans, n_max=1000, other='', verbose=False, value='itau'):
    """
    Gives the integrated optical depth for a given temperature and density. 
    The emission measure is unity. The output units are Hz.
    """

    t = str2val(temp)
    d = dens
    
    dn = fc.set_dn(trans)
    mdn = Mdn(dn)
    
    bbn = load_betabn(temp, dens, other, trans, verbose)
    n = bbn[:n_max,0]
    b = bbn[:n_max,1]
    
    if value == 'itau':
        #i = -1.069e7*dn*mdn*b*np.exp(1.58e5/(np.power(n, 2)*t))/np.power(t, 5./2.)
        i = itau_norad(n, t, b, dn, mdn)
    elif value == 'bbnMdn':
        i = b*dn*mdn
    else:
        i = b
        
    return n, i

def itau_h(temp, dens, trans, n_max=1000, other='', verbose=False, value='itau'):
    """
    Gives the integrated optical depth for a given temperature and density. 
    The emission measure is unity. The output units are Hz.
    """

    t = str2val(temp)
    d = dens
    
    dn = fc.set_dn(trans)
    mdn = Mdn(dn)
    
    bbn = load_betabn_h(temp, dens, other, trans, verbose)
    n = bbn[:,0]
    b = bbn[:,1]

    b = b[:n_max]
    n = n[:n_max]
    
    if value == 'itau':
        #i = -1.069e7*dn*mdn*b*np.exp(1.58e5/(np.power(n, 2)*t))/np.power(t, 5./2.)
        i = itau_norad(n, t, b, dn, mdn)
    elif value == 'bbnMdn':
        i = b*dn*mdn
    else:
        i = b
        
    return n, i



def load_itau_dict(dict, trans, n_max=1000, verbose=False, value='itau'):
    """
    Loads the models defined by dict.
    """
    
    data = np.zeros((len(dict['te']),2,n_max))
    
    for i,t in enumerate(dict['te']):
        
        if verbose:
            print "Trying to load model: ne={0}, te={1}, tr={2}".format(dict['ne'][i], t, dict['tr'][i])
        n, int_tau = itau(t, 
                          '{0:.4f}'.format(dict['ne'][i]), 
                          trans, 
                          n_max=n_max, 
                          other=dict['tr'][i], 
                          verbose=verbose, 
                          value=value)
        
        data[i,0] = n
        data[i,1] = int_tau
    
    te = np.asarray(map(str2val, dict['te']))
    
    return [te, dict['ne'], dict['tr'], data]

def load_itau_all(trans='alpha', n_max=1000, verbose=False, value='itau'):
    """
    Loads all the available models.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    models = glob.glob('{0}/bbn2_{1}/*'.format(LOCALDIR, trans))
    natural_sort(models)
    models = np.asarray(models)
    
    models_len = np.asarray([len(model.split('_')) for model in models])
    models_tr = sorted(models, key=lambda x: (str2val(x.split('_')[4]), 
                                                     float(x.split('_')[6]),
                                                     str2val(x.split('_')[11]) if len(x.split('_')) > 17 else 0))
    models = models_tr
    
    Te = np.zeros(len(models))
    ne = np.zeros(len(models))
    other = np.zeros(len(models), dtype='|S20')
    data = np.zeros((len(models), 2, n_max))
    
    for i,model in enumerate(models):
        if verbose:
            print model
        st = model.split('_')[4]
        Te[i] = str2val(st)
        sn = model.split('_')[6].rstrip('0')
        ne[i] = float(sn)
        if len(model.split('_')) <= 17:
            other[i] = '-'
        else:
            other[i] = '_'.join(model.split('_')[9:12])
        if verbose:
            print "Trying to load model: ne={0}, te={1}, tr={2}".format(ne[i], Te[i], other[i])
        n, int_tau = itau(st, 
                          '{0:.4f}'.format(ne[i]), 
                          trans, 
                          n_max=n_max, 
                          other=other[i], 
                          verbose=verbose, 
                          value=value)
        data[i,0] = n
        data[i,1] = int_tau
        
    return [Te, ne, other, data]

def load_itau_all_hydrogen(trans='alpha', n_max=1000, verbose=False, value='itau'):
    """
    Loads all the available models.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    models = glob.glob('{0}/H_bbn2_{1}/*'.format(LOCALDIR, trans))
    natural_sort(models)
    models = np.asarray(models)
    
    models = sorted(models, key=lambda x: (str2val(x.split('_')[4]), 
                                           float(x.split('_')[6]),
                                           str2val(x.split('_')[11]) if len(x.split('_')) > 17 else 0))
    
    Te = np.zeros(len(models))
    ne = np.zeros(len(models))#, dtype='|S20')
    other = np.zeros(len(models), dtype='|S20')
    data = np.zeros((len(models), 2, n_max))
    
    for i,model in enumerate(models):
        if verbose:
            print model
        st = model.split('_')[4]
        Te[i] = str2val(st)
        sn = model.split('_')[6].rstrip('0')
        ne[i] = float(sn)
        if len(model.split('_')) <= 17:
            other[i] = '-'
        else:
            other[i] = '_'.join(model.split('_')[9:12])
        if verbose:
            print "Trying to load model: ne={0}, te={1}, tr={2}".format(ne[i], Te[i], other[i])
        n, int_tau = itau_h(st, sn, trans, n_max=n_max, other=other[i], verbose=verbose, value=value)
        data[i,0] = n
        data[i,1] = int_tau
        
    return [Te, ne, other, data]

def load_itau_all_match(trans_out='alpha', trans_tin='beta', n_max=1000, verbose=False, value='itau'):
    """
    Loads all trans_out models that can be found in trans_tin.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    target = [f.split('/')[-1] for f in glob.glob('{0}/bbn2_{1}/*'.format(LOCALDIR, trans_tin))]
    #print target[0]
    models = ['bbn2_{0}/'.format(trans_out) + f for f in target]
    #print models[0]
    
    [Te, ne, other, data] = load_models(models, trans_out, n_max, verbose, value)
    
    return [Te, ne, other, data]

def load_itau_all_norad(trans='alpha', n_max=1000):
    """
    Loads all the available models.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    models = glob.glob('{0}/bbn/*_dat_bn_beta'.format(LOCALDIR))
    natural_sort(models)
    
    Te = np.zeros(len(models))
    ne = np.zeros(len(models))
    other = np.zeros(len(models), dtype='|S20')
    data = np.zeros((len(models), 2, n_max))
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

def load_itau_nelim(temp, dens, trad, trans, n_max=1000, verbose=False, value='itau'):
    """
    Loads models given a temperature, radiation field and an 
    upper limit for the electron density.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    models = glob.glob('{0}/bbn2_{1}/*_T_{2}_*_{3}_*'.format(LOCALDIR, trans, 
                                                             temp, trad))
    #print models
    natural_sort(models)
    models = np.asarray(models)
    
    models_len = np.asarray([len(model.split('_')) for model in models])
    models = sorted(models, key=lambda x: (str2val(x.split('_')[4]), 
                                           float(x.split('_')[6]),
                                           str2val(x.split('_')[11]) if len(x.split('_')) > 17 else 0))
    
    models = np.asarray(models)
    nes = np.asarray([float(model.split('_')[6].rstrip('0')) for model in models])
    
    # Only select those models with a density equal or lower than the specified value: dens.
    models = models[nes <= dens]
    #print models
    
    return load_models(models, trans, n_max=n_max, verbose=verbose, value=value)

def itau_norad(n, te, b, dn, mdn):
    """
    Returns the optical depth with only the approximate solution to the 
    radiative transfer problem.
    """
    
    return -1.069e7*dn*mdn*b*np.exp(1.58e5/(np.power(n, 2)*te))/np.power(te, 5./2.)

#def load_betabn(temp, dens, other=''):
    #"""
    #Loads a model for the CRRL emission.
    #"""
    
    #LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    #mod_file = '{0}/bbn/Carbon_opt_T_{1}_ne_{2}_ncrit_1.5d3_vriens_delta_500_vrinc_nmax_9900_dat_bn_beta{3}'.format(LOCALDIR, temp, dens, other)
    
    #data = np.loadtxt(mod_file)
    
    #return data

def load_betabn(temp, dens, other='', trans='alpha', verbose=False):
    """
    Loads a model for the CRRL emission.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    if other == '-' or other == '':
        model_file = 'bbn2_{0}/Carbon_opt_T_{1}_ne_{2}_ncrit_1.5d3_vriens_delta_500_vrinc_nmax_9900_datbn_beta'.format(trans, temp, dens)
        if verbose:
            print 'Will try to locate: {0}'.format(model_file)
        model_path = glob.glob('{0}/{1}'.format(LOCALDIR, model_file))[0]
    else:
        model_file = 'bbn2_{0}/Carbon_opt_T_{1}_ne_{2}_ncrit_1.5d3_{3}_vriens_delta_500_vrinc_nmax_9900_datbn_beta'.format(trans, temp, dens, other)
        if verbose:
            print 'Will try to locate: {0}'.format(model_file)
        model_path = glob.glob('{0}/{1}'.format(LOCALDIR, model_file))[0]
    
    if verbose:
        print "Loading {0}".format(model_path)
    data = np.loadtxt(model_path)
    
    return data

def load_betabn_h(temp, dens, other='', trans='alpha', verbose=False):
    """
    Loads a model for the HRRL emission.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    if other == '-' or other == '':
        model_file = 'H_bbn2_{0}/Hydrogen_opt_T_{1}_ne_{2}*_ncrit_8d2_vriens_delta_500_vrinc_nmax_9900_datbn_beta'.format(trans, temp, dens)
        if verbose:
            print 'Will try to locate: {0}'.format(model_file)
        model_path = glob.glob('{0}/{1}'.format(LOCALDIR, model_file))[0]
    else:
        model_file = 'H_bbn2_{0}/Hydrogen_opt_T_{1}_ne_{2}*_ncrit_8d2_{3}_vriens_delta_500_vrinc_nmax_9900_datbn_beta'.format(trans, temp, dens, other)
        if verbose:
            print 'Will try to locate: {0}'.format(model_file)
        model_path = glob.glob('{0}/{1}'.format(LOCALDIR, model_file))[0]
    
    if verbose:
        print "Loading {0}".format(model_path)
    data = np.loadtxt(model_path)
    
    return data

def load_bn(temp, dens, other=''):
    """
    Loads the bn values from the CRRL models.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    mod_file = '{0}/bn/Carbon_opt_T_{1}_ne_{2}_ncrit_1.5d3_vriens_delta_500_vrinc_nmax_9900_dat{3}'.format(LOCALDIR, temp, dens, other)
    
    data = np.loadtxt(mod_file)
    
    return data

def load_bn2(temp, dens, trans, other=''):
    """
    Loads the bn values from the CRRL models.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    if other == '-' or other == '':
        mod_file = 'bn2/Carbon_opt_T_{1}_ne_{2}*_ncrit_1.5d3_vriens_delta_500_vrinc_nmax_9900_dat'.format(LOCALDIR, temp, dens)
        print "Loading {0}".format(mod_file)
        mod_file = glob.glob('{0}/bn2/Carbon_opt_T_{1}_ne_{2}*_ncrit_1.5d3_vriens_delta_500_vrinc_nmax_9900_dat'.format(LOCALDIR, temp, dens))[0]
    else:
        mod_file = 'bn2/Carbon_opt_T_{1}_ne_{2}*_ncrit_1.5d3_{3}_vriens_delta_500_vrinc_nmax_9900_dat'.format(LOCALDIR, temp, dens, other)
        print "Loading {0}".format(mod_file)
        mod_file = glob.glob('{0}/bn2/Carbon_opt_T_{1}_ne_{2}*_ncrit_1.5d3_{3}_vriens_delta_500_vrinc_nmax_9900_dat'.format(LOCALDIR, temp, dens, other))[0]
    
    print "Loaded {0}".format(mod_file)
    data = np.loadtxt(mod_file)
    
    return data
    
    
    #mod_file = '{0}/bn/Carbon_opt_T_{1}_ne_{2}_ncrit_1.5d3_vriens_delta_500_vrinc_nmax_9900_dat{3}'.format(LOCALDIR, temp, dens, other)
    
    #data = np.loadtxt(mod_file)
    
    #return data
    
def load_models(models, trans, n_max=1000, verbose=False, value='itau'):
    """
    Loads the models in backwards compatible mode.
    It will sort the models by Te, ne and Tr.
    """

    models = np.asarray(models)
    
    models_len = np.asarray([len(model.split('_')) for model in models])
    models = sorted(models, key=lambda x: (str2val(x.split('_')[4]), 
                                                   float(x.split('_')[6]),
                                                   str2val(x.split('_')[11]) if len(x.split('_')) > 17 else 0))
        
    Te = np.zeros(len(models))
    ne = np.zeros(len(models))
    other = np.zeros(len(models), dtype='|S20')
    data = np.zeros((len(models), 2, n_max))
    
    for i,model in enumerate(models):
        if verbose:
            print model
        st = model.split('_')[4]
        Te[i] = str2val(st)
        sn = model.split('_')[6].rstrip('0')
        ne[i] = float(sn)
        if len(model.split('_')) <= 17:
            other[i] = '-'
        else:
            other[i] = '_'.join(model.split('_')[9:12])
        if verbose:
            print "Trying to load model: ne={0}, te={1}, tr={2}".format(ne[i], Te[i], other[i])
        n, int_tau = itau(st, sn, trans, n_max=n_max, other=other[i], verbose=verbose, value=value)
        data[i,0] = n
        data[i,1] = int_tau
        
    return [Te, ne, other, data]

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
            e = Decimal(-h.value*freq[i]*1e6/(k_B.value*t))
            exp = Decimal(e).exp()#Decimal(np.exp(-h.value*freq[i]*1e6/(k_B.value*t)))
            beta[i] = float((Decimal(1) - bnn*exp)/(Decimal(1) - exp))
        
    return np.array([bn[:-1,0], beta*bn[:-1,1]])

def make_betabn2(temp, dens, trans, n_max=1000, other=''):
    """
    """
    
    t = str2val(temp)
    d = dens
    
    dn = fc.set_dn(trans)
    bn = load_bn2(temp, dens, other=other)
    specie, trans, n, freq = fc.make_line_list('CI', bn[-1,0]+1, dn)
    # Cut bn first
    bn = bn[:n_max]#bn[:np.where(bn[:,0]==n_max)[0]]
    freq = freq[np.where(n==bn[0,0])[0]:np.where(n==bn[-1,0])[0]]
    
    beta = np.empty(len(freq))
    for i in xrange(len(freq)):
        if i < len(freq)-dn:
            #bnn = np.divide(bn[i+dn,-1], bn[i,-1])
            bnn = Decimal(bn[i+dn,-1]) / Decimal(bn[i,-1])
            e = Decimal(-h.value*freq[i]*1e6/(k_B.value*t))
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
    
    try:
        aux = map(float, str.split('d'))
        val = aux[0]*np.power(10., aux[1])
    except ValueError:
        val = 0
    
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
    
def valid_ne(trans):
    """
    Checks all the available models and lists the available ne values.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    models = glob.glob('{0}/bbn2_{1}/*'.format(LOCALDIR, trans))
    natural_sort(models)
    models = np.asarray(models)
    
    models_len = np.asarray([len(model.split('_')) for model in models])
    #models_tr = models[models_len>17]
    #print models_tr[0].split('_')[11], models_tr[0].split('_')[4], models_tr[0].split('_')[6]
    models = sorted(models, key=lambda x: (str2val(x.split('_')[4]), 
                                           float(x.split('_')[6]),
                                           str2val(x.split('_')[11]) if len(x.split('_')) > 17 else 0))
    ne = np.asarray([float(model.split('_')[6].rstrip('0')) for model in models])
    
    return np.unique(ne)