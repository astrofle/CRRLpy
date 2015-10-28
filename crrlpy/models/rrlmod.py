#!/usr/bin/env python

"""
.. module:: rrlmod
   :platform: Unix
   :synopsis: RRL model tools.

.. moduleauthor:: Pedro Salas <psalas@strw.leidenuniv.nl>

"""

#__docformat__ = 'reStructuredText'

from __future__ import division

import numpy as np
import os
import glob
import re

import astropy.units as u

from crrlpy import frec_calc as fc
from astropy.constants import h, k_B, c, m_e, Ryd, e
from astropy.analytic_functions import blackbody_nu
from crrlpy.crrls import natural_sort, best_match_indx2, f2n, n2f
from mpmath import mp
mp.dps = 50 

LOCALDIR = os.path.dirname(os.path.realpath(__file__))

def broken_plaw(nu, nu0, T0, alpha1, alpha2):
    """
    Defines a broken power law.
    
    .. math::
    
       T(\\nu) = T_{0}\\left(\\dfrac{\\nu}{\\nu_{0}}\\right)^{\\alpha_1}\\mbox{ if }\\nu<\\nu_{0}
       
       T(\\nu) = T_{0}\\left(\\dfrac{\\nu}{\\nu_{0}}\\right)^{\\alpha_2}\\mbox{ if }\\nu\\geq\\nu_{0}
    
    :param nu: Frequency.
    :param nu0: Frequency at which the power law breaks.
    :param T0: Value of the power law at nu0.
    :param alpha1: Index of the power law for nu<nu0.
    :param alpha2: Index of the power law for nu>=nu0.
    :returns: Broken power law evalueated at nu. 
    """
        
    low = plaw(nu, nu0, T0, alpha1) * (nu < nu0)
    hgh = plaw(nu, nu0, T0, alpha2) * (nu >= nu0)
    
    return low + hgh
    
def eta(freq, Te, ne, nion, Z, Tr, trans, n_max=1500):
    """
    Returns the correction factor for the Planck function.
    """
    
    kl = kappa_line_lte(Te, ne, nion, Z, Tr, trans, n_max=1500)
    kc = kappa_cont(freq, Te, ne, nion, Z)
    
    return (kc + kl*bni)/(kc + kl*bnf*bnfni)

def I_Bnu(specie, Z, n, Inu_funct, *args):
    """
    Calculates the product :math:`B_{n+\\Delta n,n}I_{\\nu}` to 
    compute the line broadening due to a radiation field :math:`I_{\\nu}`.
    
    :param specie: Atomic specie to calculate for.
    :param n: Principal quantum number at which to evaluate :math:`\\frac{2}{\\pi}\\sum_{\\Delta n}B_{n+\\Delta n,n}I_{n+\\Delta n,n}(\\nu)`.
    :param Inu_funct: Function to call and evaluate :math:`I_{n+\\Delta n,n}(\\nu)`. It's first argument must be the frequency.
    :param *args: Arguments to Inu_funct. The frequency must be left out. \
    The frequency will be passed internally in units of MHz. Use the same \
    unit when required. :paramref:`Inu_funct` must take the frequency as first parameter.
    
    """
    
    cte = 2.*np.pi**2.*e.gauss**2./(h.cgs.value*m_e.cgs.value*c.cgs.value**2.*Ryd.to('1/cm')*Z**2.)

    Inu = np.empty((len(n), 5))
    BninfInu = np.empty((len(n), 5))
    nu = np.empty((len(n), 5))
    
    for dn in range(1,6):
        nu[:,dn-1] = n2f(n, specie+fc.set_trans(dn))
        Inu[:,dn-1] = Inu_funct(nu[:,dn-1]*1e6, *args).cgs.value
        BninfInu[:,dn-1] = cte.cgs.value*n**6./(dn*(n + dn)**2.)*Mdn(dn)*Inu[:,dn-1]
    
    return 2./np.pi*BninfInu.sum(axis=1)

def I_broken_plaw(nu, Tr, nu0, alpha1, alpha2):
    """
    Returns the blackbody function evaluated at nu. 
    As temperature a broken power law is used.
    The power law shape has parameters: Tr, nu0, alpha1 and alpha2.
    
    :param nu: Frequency. (Hz) or astropy.units.Quantity_
    :type nu: (Hz) or astropy.units.Quantity_
    :param Tr: Temperature at nu0. (K) or astropy.units.Quantity_
    :param nu0: Frequency at which the spectral index changes. (Hz) or astropy.units.Quantity_
    :param alpha1: spectral index for :math:`\\nu<\\nu_0`
    :param alpha2: spectral index for :math:`\\nu\\geq\\nu_0`
    :returns: Specific intensity in :math:`\\rm{erg}\\,\\rm{cm}^{-2}\\,\\rm{Hz}^{-1}\\,\\rm{s}^{-1}\\,\\rm{sr}^{-1}`. See `astropy.analytic_functions.blackbody.blackbody_nu`__
    :rtype: astropy.units.Quantity_
    
    .. _astropy.units.Quantity: http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html#astropy.units.Quantity
    __ blackbody_
    .. _blackbody: http://docs.astropy.org/en/stable/api/astropy.analytic_functions.blackbody.blackbody_nu.html#astropy.analytic_functions.blackbody.blackbody_nu
    """
    
    Tbpl = broken_plaw(nu, nu0, Tr, alpha1, alpha2)
    
    bnu_bpl = blackbody_nu(nu, Tbpl)
    
    return bnu_bpl

def I_cont(nu, Te, tau, I0, unitless=False):
    """
    Computes the specific intensity due to a blackbody at temperature :math:`T_{e}` and optical depth :math:`\\tau`. It considers that there is 
    background radiation with :math:`I_{0}`.
    
    :param nu: Frequency.
    :type nu: (Hz) or astropy.units.Quantity_
    :param Te: Temperature of the source function. (K) or astropy.units.Quantity_
    :param tau: Optical depth of the medium.
    :param I0: Specific intensity of the background radiation. Must have units of erg / (cm2 Hz s sr) or see :paramref:`unitless`.
    :param unitless: If True the return 
    :returns: The specific intensity of a ray of light after traveling in an LTE \
    medium with source function :math:`B_{\\nu}(T_{e})` after crossing an optical \
    depth :math:`\\tau_{\\nu}`. The units are erg / (cm2 Hz s sr). See `astropy.analytic_functions.blackbody.blackbody_nu`__
    
    __ blackbody_
    .. _blackbody: http://docs.astropy.org/en/stable/api/astropy.analytic_functions.blackbody.blackbody_nu.html#astropy.analytic_functions.blackbody.blackbody_nu
    """
    
    bnu = blackbody_nu(nu, Te)
    
    if unitless:
        bnu = bnu.cgs.value
        
    return bnu*(1. - np.exp(-tau)) + I0*np.exp(-tau)

def I_external(nu, Tbkg, Tff, tau_ff, Tr, nu0=100e6*u.MHz, alpha=-2.6):
    """
    This method is equivalent to the IDL routine 
    
    :param nu: Frequency. (Hz) or astropy.units.Quantity_
    
    .. _astropy.units.Quantity: http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html#astropy.units.Quantity
    """
    
    if Tbkg.value != 0:
        bnu_bkg = blackbody_nu(nu, Tbkg)
    else:
        bnu_bkg = 0
    
    if Tff.value != 0:
        bnu_ff = blackbody_nu(nu, Tff)
        exp_ff = (1. - np.exp(-tau_ff))
    else:
        bnu_ff = 0
        exp_ff = 0
        
    if Tr.value != 0:
        Tpl = plaw(nu, nu0, Tr, alpha)#Tr*np.power(nu/nu0, alpha)
        bnu_pl = blackbody_nu(nu, Tpl)
    else:
        bnu_pl = 0
    
    return bnu_bkg + bnu_ff*exp_ff + bnu_pl

def I_total(nu, Te, tau, I0, eta):
    """
    """
    
    bnu = blackbody_nu(nu, Te)
    
    exp = np.exp(-tau)
    
    return bnu*eta*(1. - exp) + I0*exp

def itau(temp, dens, line, n_min=5, n_max=1000, other='', verbose=False, value='itau'):
    """
    Gives the integrated optical depth for a given temperature and density. 
    The emission measure is unity. The output units are Hz.
    
    :param temp: Electron temperature. Must be a string of the form '8d1'.
    :param dens: Electron density. Float
    :param line: Line to load models for.
    :param n_min: Minimum n value to include in the output. Int Default 1
    :param n_max: Maximum n value to include in the output. Int Default 1500, Maximum allowed value 9900
    :param other: String to search for different radiation fields and others.
    :param verbose: Verbose output? Bool
    :param value: ['itau'|'bbnMdn'|none] Value to output. itau will output the integrated optical depth.
    bbnMdn will output the :math:`\\beta_{n,n^{\\prime}}b_{n}` times the oscillator strenght :math:`M(\\Delta n)`.
    :returns: The principal quantum number and its asociated value.
    """

    t = str2val(temp)
    d = dens
    
    dn = fc.set_dn(line)
    mdn = Mdn(dn)
    
    bbn = load_betabn(temp, dens, other, line, verbose)
    nimin = best_match_indx2(n_min, bbn[:,0])
    nimax = best_match_indx2(n_max, bbn[:,0])
    n = bbn[nimin:nimax,0]
    b = bbn[nimin:nimax,1]
    
    if value == 'itau':
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

def j_line_lte(n, ne, nion, Te, Z, trans):
    """
    """
    
    trans = fc.set_name(trans)
    
    Aninf = np.loadtxt('{0}/rates/einstein_Anm_{1}.txt'.format(LOCALDIR, trans))
    cte = h/4./np.pi
    Nni = level_pop_lte(n, ne, nion, Te, Z)
    lc = line_center(nu)
    
    return cte*Aninf[:,2]*Nni*lc

def kappa_cont(freq, Te, ne, nion, Z):
    """
    Computes the absorption coefficient for the free-free process.
    """
    
    nu = freq.to('GHz').value
    v = 0.65290+2./3.*np.log10(nu) - np.log10(Te.to('K').value)
    
    kc = np.zeros(len(nu))
    
    #mask_1 = v <= -2.6
    mask_2 = (v > -5) & (v <= -0.25)
    mask_3 = v > -0.25
    
    #kc[mask_1] = 6.9391e-8*np.power(Z, 2)*ne*nion*np.power(nu[mask_1], -2.)* \
                 #np.power(1e4/Te.to('K').value, 1.5)* \
                 #(4.6950 - np.log10(Z) + 1.5*np.log(Te.to('K').value/1e4) - np.log10(nu[mask_1]))
    
    v_2 = v[mask_2]
    log10I_m0 = -1.232644*v_2 + 0.098747
    kc[mask_2] = kappa_cont_base(nu[mask_2], Te.to('K').value, ne.cgs.value, 
                                 nion.cgs.value, Z)*np.power(10., log10I_m0)
    
    v_3 = v[mask_3]
    log10I_m0 = -1.084191*v_3 + 0.135860
    kc[mask_3] = kappa_cont_base(nu[mask_3], Te.to('K').value, ne.cgs.value, 
                                 nion.cgs.value, Z)*np.power(10., log10I_m0)
    
    #if v < -2.:
        #kc = 6.9391e-8*np.power(Z, 2)*ne*nion*np.power(nu, -2.)*np.power(1e4/Te, 1.5)* \
             #(4.6950 - np.log10(Z) + 1.5*np.log(Te/1e4) - np.log10(nu))
    #elif v >= -2. and v <= -0.25:
        #log10I_m0 = -1.232644*v + 0.098747
    #elif v > -0.25:
        #log10I_m0 = -1.084191*v + 0.135860
    
    #if v >= -2.:
        #kc = 4.6460/np.power(nu, 7./3.)/np.power(Te, 1.5)* \
            #(np.exp(4.7993e-2*nu/Te) - 1.)* \
            #np.exp(aliv/np.log10(np.exp(1)))#*np.exp(-h*freq/k_B/Te)
        
    return kc*u.pc**-1*np.exp(-h.cgs.value*nu/k_B.cgs.value/Te.cgs.value)

def kappa_cont_base(nu, Te, ne, nion, Z):
    """
    
    """
    
    return 4.6460/np.power(nu, 7./3.)/np.power(Te, 1.5)* \
            (np.exp(4.7993e-2*nu/Te) - 1.)*np.power(Z, 8./3.)*ne*nion

def kappa_line(Te, ne, nion, Z, Tr, trans, n_max=1500):
    """
    Computes the line absorption coefficient for CRRLs between levels :mat:`n_{i}` and :math:`n_{f}`, :math:`n_{i}>n_{f}`.
    This can only go up to :math:`n_{\\rm{max}}` 1500 because of the tables used for the Einstein Anm coefficients.
    
    :param Te: Electron temperature of the gas. (K)
    :type Te: float
    :param ne: Electron density. (:math:`\\mbox{cm}^{-3}`)
    :type ne: float
    :param nion: Ion density. (:math:`\\mbox{cm}^{-3}`)
    :type nion: float
    :param Z: Electric charge of the atoms being considered.
    :type Z: int
    :param Tr: Temperature of the radiation field felt by the gas. This specifies the temperature of the field at 100 MHz. (K)
    :type Tr: float
    :param trans: Transition for which to compute the absorption coefficient.
    :type trans: string
    :param n_max: Maximum principal quantum number to include in the output.
    :type n_max: int<1500
    :returns: 
    :rtype: array
    """
    
    cte = np.power(c, 2.)/(16.*np.pi)*np.power(np.power(h, 2)/(2.*np.pi*m_e*k_B), 3./2.)
    
    bn = load_bn2(val2str(Te), ne, other='case_diffuse_{0}'.format(val2str(Tr)))
    bn = bn[:np.where(bn[:,0] == n_max)[0]]
    Anfni = np.loadtxt('{0}/rates/einstein_Anm_{1}.txt'.format(LOCALDIR, trans))
    
    # Cut the Einstein Amn coefficients table to match the bn values
    i_bn_i = best_match_indx2(bn[0,0], Anfni[:,1])
    i_bn_f = best_match_indx2(bn[-1,0], Anfni[:,0])
    Anfni = Anfni[i_bn_i:i_bn_f+1]
    
    ni = Anfni[:,0]
    nf = Anfni[:,1]
    
    omega_ni = 2*np.power(ni, 2)
    omega_i = 1.
    
    xi_ni = xi(ni, Te, Z)
    xi_nf = xi(nf, Te, Z)
    
    exp_ni = np.exp(xi_ni.value)
    exp_nf = np.exp(xi_nf.value)
    
    #print len(Anfni), len(bn[1:,-1]), len(bn[:-1,-1]), len(omega_ni[:]), len(ni), len(exp_ni), len(exp_nf)
    kl = cte.value/np.power(Te, 3./2.)*ne*nion*Anfni[:,2]*omega_ni[:]/omega_i*(bn[1:,-1]*exp_ni - bn[:-1,-1]*exp_nf)
    
    return kl

def kappa_line_lte(nu, Te, ne, nion, Z, Tr, line, n_min=1, n_max=1500):
    """
    Returns the line absorption coefficient under LTE conditions.
    
    :param nu: Frequency. (Hz)
    :type nu: array
    :param Te: Electron temperature of the gas. (K)
    :type Te: float
    :param ne: Electron density. (:math:`\\mbox{cm}^{-3}`)
    :type ne: float
    :param nion: Ion density. (:math:`\\mbox{cm}^{-3}`)
    :type nion: float
    :param Z: Electric charge of the atoms being considered.
    :type Z: int
    :param Tr: Temperature of the radiation field felt by the gas. This specifies the temperature of the field at 100 MHz. (K)
    :type Tr: float
    :param trans: Transition for which to compute the absorption coefficient.
    :type trans: string
    :param n_max: Maximum principal quantum number to include in the output.
    :type n_max: int<1500
    :returns: 
    :rtype: array
    """
    
    ni = f2n(nu.to('MHz').value, line, n_max) + 1.
    
    trans = fc.set_name(line)
    
    cte = (np.power(c, 2.)/(8.*np.pi))
    Aninf = np.loadtxt('{0}/rates/einstein_Anm_{1}.txt'.format(LOCALDIR, trans))
    Aninf = Aninf[np.where(Aninf[:,1] == n_min)[0]:np.where(Aninf[:,1] == n_max)[0]]
    
    exp = np.exp(-h*nu/k_B/Te)
    Nni = level_pop_lte(ni, ne, nion, Te, Z)
    
    return cte/np.power(nu, 2.)*Nni*Aninf[:,2]*(1. - exp)#*np.power(Aninf[:,0]/Aninf[:,1], 2.)
    

def level_pop_lte(n, ne, nion, Te, Z):
    """
    Returns the level population of level n.
    The return has units of :math:`\\mbox{cm}^{-3}`.
    """
    
    omega_ni = 2*np.power(n, 2)
    omega_i = 1.
    
    xi_n = xi(n, Te, Z)
    
    exp_xi_n = np.exp(xi_n.value)
    
    Nn = ne*nion*np.power(np.power(h, 2.)/(2*np.pi*m_e*k_B*Te), 1.5)*omega_ni/omega_i/2.*exp_xi_n
    
    return Nn

def load_itau_dict(dict, trans, n_min=5, n_max=1000, verbose=False, value='itau'):
    """
    Loads the models defined by dict.
    """
    
    data = np.zeros((len(dict['Te']),2,n_max-n_min))
    
    for i,t in enumerate(dict['Te']):
        
        if verbose:
            print "Trying to load model: ne={0}, Te={1}, Tr={2}".format(dict['ne'][i], t, dict['Tr'][i])
        n, int_tau = itau(t, 
                          dict['ne'][i], 
                          trans,
                          n_min=n_min,
                          n_max=n_max, 
                          other=dict['Tr'][i], 
                          verbose=verbose, 
                          value=value)
        
        data[i,0] = n
        data[i,1] = int_tau
    
    te = np.asarray(map(str2val, dict['Te']))
    
    return [te, dict['ne'], dict['Tr'], data]

def load_itau_all(trans='CIalpha', n_min=5, n_max=1000, verbose=False, value='itau'):
    """
    Loads all the available models for Carbon.
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
    data = np.zeros((len(models), 2, n_max-n_min))
    
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
        n, int_tau = itau(st, ne[i], trans, n_min=n_min, n_max=n_max, 
                          other=other[i], verbose=verbose, 
                          value=value)
        data[i,0] = n
        data[i,1] = int_tau
        
    return [Te, ne, other, data]

def load_itau_all_hydrogen(trans='alpha', n_max=1000, verbose=False, value='itau'):
    """
    Loads all the available models for Hydrogen.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    models = glob.glob('{0}/H_bbn2_{1}/*'.format(LOCALDIR, trans))
    natural_sort(models)
    models = np.asarray(models)
    
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
        n, int_tau = itau_h(st, sn, trans, n_max=n_max, other=other[i], verbose=verbose, value=value)
        data[i,0] = n
        data[i,1] = int_tau
        
    return [Te, ne, other, data]

def load_itau_all_match(trans_out='alpha', trans_tin='beta', n_max=1000, verbose=False, value='itau'):
    """
    Loads all trans_out models that can be found in trans_tin. This is useful when analyzing line ratios.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    target = [f.split('/')[-1] for f in glob.glob('{0}/bbn2_{1}/*'.format(LOCALDIR, trans_tin))]
    models = ['bbn2_{0}/'.format(trans_out) + f for f in target]
    
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

        n, int_tau = itau(st, sn, trans, n_max=1000, other=other[i])

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

def load_betabn(temp, dens, other='', trans='CIalpha', verbose=False):
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
        model_file = 'H_bbn2_{0}/Hydrogen_opt_T_{1}_ne_{2}_ncrit_8d2_vriens_delta_500_vrinc_nmax_9900_datbn_beta'.format(trans, temp, dens)
        if verbose:
            print 'Will try to locate: {0}'.format(model_file)
        model_path = glob.glob('{0}/{1}'.format(LOCALDIR, model_file))[0]
    else:
        model_file = 'H_bbn2_{0}/Hydrogen_opt_T_{1}_ne_{2}_ncrit_8d2_{3}_vriens_delta_500_vrinc_nmax_9900_datbn_beta'.format(trans, temp, dens, other)
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

def load_bn2(temp, dens, other=''):
    """
    Loads the bn values from the CRRL models.
    """
    
    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    
    if other == '-' or other == '':
        mod_file = 'bn2/Carbon_opt_T_{1}_ne_{2}_ncrit_1.5d3_vriens_delta_500_vrinc_nmax_9900_dat'.format(LOCALDIR, temp, dens)
        print "Loading {0}".format(mod_file)
        mod_file = glob.glob('{0}/bn2/Carbon_opt_T_{1}_ne_{2}*_ncrit_1.5d3_vriens_delta_500_vrinc_nmax_9900_dat'.format(LOCALDIR, temp, dens))[0]
    else:
        mod_file = 'bn2/Carbon_opt_T_{1}_ne_{2}_ncrit_1.5d3_{3}_vriens_delta_500_vrinc_nmax_9900_dat'.format(LOCALDIR, temp, dens, other)
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

def make_betabn(line, temp, dens, n_min=5, n_max=1000, other=''):
    """
    """
    
    t = str2val(temp)
    d = str2val(dens)
    
    bn = load_bn(temp, dens, other=other)
    line, n, freq = fc.make_line_list(line, n_min=n_min, n_max=n_max)
    
    # Cut bn first
    bn = bn[np.where(bn[:,0]==n_min)[0]:np.where(bn[:,0]==n_max)[0]]
    freq = freq[np.where(n==bn[0,0])[0]:np.where(n==bn[-1,0])[0]]
    
    beta = np.empty(len(freq))
    
    for i in xrange(len(freq)):
        if i < len(freq)-dn:

            bnn = Decimal(bn[i+dn,-1]) / Decimal(bn[i,-1])
            e = Decimal(-h.value*freq[i]*1e6/(k_B.value*t))
            exp = Decimal(e).exp()
            beta[i] = float((Decimal(1) - bnn*exp)/(Decimal(1) - exp))
        
    return np.array([bn[:-1,0], beta*bn[:-1,1]])

def make_betabn2(line, temp, dens, n_min=5, n_max=1000, other=''):
    """
    """
    
    t = str2val(temp)
    d = dens
    
    dn = fc.set_dn(line)
    bn = load_bn2(temp, dens, other=other)
    line, n, freq = fc.make_line_list(line, n_min=n_min, n_max=bn[-1,0]+1)
    
    # Cut bn first
    bn = bn[np.where(bn[:,0]==n_min)[0]:np.where(bn[:,0]==n_max)[0]]
    #bn = bn[n_min:n_max]
    freq = freq[np.where(n==bn[0,0])[0]:np.where(n==bn[-1,0])[0]]
    
    beta = np.empty(len(freq))
    
    for i in xrange(len(freq)):
        if i < len(freq)-dn:
          
            bnn = Decimal(bn[i+dn,-1]) / Decimal(bn[i,-1])
            e = Decimal(-h.value*freq[i]*1e6/(k_B.value*t))
            exp = Decimal(e).exp()
            beta[i] = float((Decimal(1) - bnn*exp)/(Decimal(1) - exp))
        
    return np.array([bn[:-1,0], beta*bn[:-1,1]])
    
def Mdn(dn):
    """
    Gives the :math:`M(\\Delta n)` factor for a given :math:`\\Delta n`.
    ref. Menzel (1968)
    
    :param dn: :math:`\\Delta n`.
    :returns: :math:`M(\\Delta n)`
    :rtype: float
    
    :Example:
    
    >>> Mdn(1)
    0.1908
    >>> Mdn(5)
    0.001812
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

def plaw(x, x0, y0, alpha):
    """
    Returns a power law.
    
    .. math::
    
       y(x)=y_0\\left(\\frac{x}{x_0}\\right)^{alpha}
       
    :param x: x values for which to compute the power law.
    :type x: float or array like
    :param x0: x value for which the power law has amplitude :paramref:`y0`.
    :type x0: float
    :param y0: Amplitude of the power law at :paramref:`x0`.
    :type y0: float
    :param alpha: Index of the power law.
    :type alpha: float
    :returns: A power law of index :paramref:`alpha` evaluated at :paramref:`x`, with amplitude :paramref:`y0` at :paramref:`x0`.
    :rtype: float or array
    """
    
    return y0*np.power(x/x0, alpha)

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

def xi(n, Te, Z):
    """
    """
    
    return np.power(Z, 2.)*h*c*Ryd/(k_B*np.power(n, 2)*Te)