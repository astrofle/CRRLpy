#!/usr/bin/env python
"""
Equations from Goldsmith et al. (2012) 
and Goicoechea et al. (2015) to model 
the [CII] line emission.
"""

import numpy as np

from scipy.special import expi
from astropy import units as u
from astropy import constants as ac


gu = 4.                      # Statistical weigths.
gl = 2.
tstar = 91.25*u.K            # Energy above ground level of the [CII] transition in K.
aul = 2.36e-6/u.s            # Einstein A.
nu_cii = 1900.5369*u.GHz     # Frequency of the [CII] line.
nu_13cii21 = 1900.4661*u.GHz # Frequency of the strongest [13CII] line.


def beta_esc(tau):
    """
    Eq. (5) Goldsmith et al. (2012)
    """
    
    return (1. - np.exp(-tau))/tau


def beta_broad(tau_):
    """
    Eq. (4) in Hacar et al. (2016)
    """

    return 1./np.sqrt(np.log(2.))*np.sqrt( np.log( tau_ / np.log(2./(np.exp(-tau_)+1)) ) )


def col_dens(tex, tau, dv):
    """
    Eq. (6) Goicoechea et al. (2015)
    
    :param dv: Line width in km s-1.
    """
    
    return 1.5e17*(1. + 2.*np.exp(-tstar/tex))/(1. - np.exp(-tstar/tex))*tau*dv


def col_dens_err(tex, dtex, tau, dtau, fwhm, dfwhm):
    """
    """
    
    cte = 1.5e17
    r = (1. + 2.*np.exp(-tstar/tex))/(1. - np.exp(-tstar/tex))
    fac1 = (3.*a*np.exp(tstar/tex))/(tex**2.*(-1. + np.exp(tstar/tex))**2.)*dtex
    fac2 = r*fwhm*dtau
    fac3 = r*dfwhm*tau
    err = cte*np.sqrt(fac1**2. + fac2**2. + fac3**2.)
    
    return err


def cul(tgas, ne, nh, nh2):
    """
    Eq. (3) Goldsmith et al. (2012)
    """
    
    try:
        tgas = tgas.to('K').value
    except:
        pass
    
    gamma_e = gamma_e(tgas, method='PG')*u.cm**3/u.s
    gamma_h = gamma_h(tgas, method='PGe')*u.cm**3/u.s
    gamma_h2 = gamma_h2(tgas, method='PG')*u.cm**3/u.s
    
    return gamma_e*ne + gamma_h*nh + gamma_h2*nh2


def gamma_e(te, method='FS'):
    """
    Computes the de-excitation rate of the CII atom due to collisions with electrons.
    
    :param Te: Electron temperature.
    :type Te: float
    :returns: The collisional de-excitation rate in units of cm-3 s-1.
    :rtype: float
    """
    
    rates = {'FS': lambda t: 4.51e-6*np.power(t, -0.5),
             'PG': lambda t: 8.7e-8*np.power(t/2000., -0.37)}
    
    return rates[method](te) #4.51e-6*np.power(Te, -0.5)


def gamma_h(te, method='FS'):
    """
    Computes the de-excitation rate of the CII atom due to collisions with hydrogen atoms.
    
    :param Te: Electron temperature.
    :type Te: float
    :returns: The collisional de-excitation rate in units of cm-3 s-1.
    :rtype: float
    """
    
    rates = {'FS': lambda t: 5.8e-10*np.power(t, 0.02),
             'PG': lambda t: 7.6e-10*np.power(t/100., 0.14),
             'PGe': lambda t: 4e-11*(16. + 0.35*np.power(t, 0.5) + 48./t)}
    
    return rates[method](te)


def gamma_h2(te, method='TH'):
    """
    """
    
    rates = {'TH': lambda t: 3.1e-10*np.power(t, 0.1),
             'PG': lambda t: 3.8e-10*np.power(t/100., 0.14)}
    
    return rates[method](te)


def tau0(ncii, dv):
    """
    Eq. (14) Goldsmith et al. (2012)
    """
    
    g = gu/gl
    cte = ac.c**3.*aul/(8.*np.pi*nu_cii**3.)*g
    
    return cte*ncii/dv


def tau_tex(tex, tau0_):
    """
    Eq. (15) Goldsmith et al. (2012)
    """
    
    g = gu/gl
    
    return tau0_*(1. - np.exp(-tstar/tex))/(1. + g*np.exp(-tstar/tex))


def tau_tkin(tkin, tbg, tau0_, cul_):
    """
    Eq. (15) Goldsmith et al. (2012)
    """
    
    beta = beta_esc(tau_)
    x = cul_/(beta*aul)
    g = gu/gl
    k = np.exp(tstar/tkin)
    gbg_ = gbg(tbg)
    
    return tau0_*(x*(k - 1.) + k)/(x*(k + g) + k*(1. + gbg_*(1. + g)))


def gbg(tbg):
    """
    Eq. (8) Goldsmith et al. (2012)
    """
    
    if tbg != 0:
        g = 1./(np.exp(tstar/tbg) - 1.)
    else:
        g = 0.
        
    return g


def compute_ta(tkin, ne, nh, nh2, Ncii, dv, tbg, tex_min=1, tex_max=500, dtex=1):
    """
    """

    cul_ = cul(tkin, ne, nh, nh2)
    tau0_ = tau0(Ncii, dv)
    tex_eq = find_tex(tkin, cul_, tau0_, tex_min=tex_min, tex_max=tex_max, dtex=dtex).to('K').value
    tau_eq = tau_tex(tex_eq*u.K, tau0_)
    beta_eq = beta_esc(tau_eq)
    ta_ = ta(tex_eq*u.K, tbg, tau_eq).to('K')
    beta_dop = beta_broad(tau_eq)
    
    return ta_, tau_eq.cgs, tex_eq, beta_dop
    

def ta(tex, tbg, tau_):
    """
    Eq. (18) Goldsmith et al. (2012)
    """

    gbg_ = gbg(tbg)

    return tstar*(1./(np.exp(tstar/tex) - 1.) - gbg_)*(1. - np.exp(-tau_))


def ta_thick(tkin, tbg, cul_, tau_):
    """
    Eq. (33) Goldsmith et al. (2012)
    """
    
    x = (cul_)/aul/beta_esc(tau_)
    k = np.exp(tstar/tkin)
    gbg_ = gbg(tbg)
    
    return tstar*x*(1. - gbg_*(k - 1.))/(x*(k - 1) + k)


def ta_thin(tkin, ncii, dv, cul_, nu=nu_cii):
    """
    Eq. (26) Goldsmith et al. (2012)
    """
    
    #cte = 3.43e-16
    cte = ac.h*ac.c**3./(8*np.pi*ac.k_B*nu**2.)*aul
    
    return cte/(1. + 0.5*np.exp(tstar/tkin)*(1. + aul/cul_))*ncii/dv


def tkin2tex_ratio(tkin, tbg, cul_, tau_):
    """
    Eq. (13) Goldsmith et al. (2012)
    """
    
    x = (cul_*tau_)/aul
    k = np.exp(tstar/tkin)
    gbg_ = gbg(tbg)
    
    ratio = (x + 1. + gbg_)/(x + gbg_*k)
    
    return ratio


def J(tbg):
    """
    Goicoechea et al. (2015)
    """
    
    return tstar/(np.exp(tstar/tbg) - 1.)


def tex_cii(tcii, tbg):
    """
    Eq. (5) Goicoechea et al. (2015)
    """
    
    return tstar/np.log(1. + tstar/(tcii + J(tbg)))


def left_hand_side_tau(tau):
    """
    Left hand side of Eq. (4) Goicoechea et al. (2015)
    """
    
    return (1. - np.exp(-tau))/tau


def compute_tau(tcii_peak, tcii_iso_peak, ciso_abu=67., tau_low=0.01, tau_hgh=7.):
    """
    Computes the [CII] optical depth.
    """
    
    i13cii_2_1 = 0.625
    
    tau_arr = np.arange(tau_low, tau_hgh, 0.01)
    
    rhs = (i13cii_2_1*tcii_peak/tcii_iso_peak)/ciso_abu
    
    tau_indx = np.argmin(abs(rhs - left_hand_side_tau(tau_arr)))
    tau_val = tau_arr[tau_indx]
    
    return tau_val


def i2tadv(i_cii):
    """
    Intensity in cgs units to K km s-1.
    Eq. (20) Goldsmith et al. (2012)
    """
    
    # Hz -> km s-1
    i_cii_k = i_cii*ac.c/nu_cii
    # erg -> K
    i_cii_k = i_cii_k/(2.*ac.k_B*np.power(nu_cii/ac.c, 2.))
    
    return i_cii_k


def find_tex(tkin, cul_, tau0_, tex_min=30, tex_max=500, dtex=5):
    """
    Solve the two level problem for [CII] and find the excitation temperature, tex.
    """
    
    tex = np.arange(tex_min, tex_max, dtex)*u.K
    
    tau_ = tau_tex(tex, tau0_)
    beta = beta_esc(tau_)
    rhs = lambda tex : (cul_ + beta*aul)/(cul_*np.exp(-tstar/tkin))
    lhs = lambda tex : np.exp(tstar/tex)
    
    eq_idx = np.argmin(abs(lhs(tex) - rhs(tex)))
    tex_eq = tex[eq_idx]
    
    if eq_idx == 0 or eq_idx == len(tex):
        print("Tex range too small.")
    
    return tex_eq


def tex_levelpop(nl, nu):
    """
    Excitation temperature from the density 
    of atoms in the upper and lower states.
    Eq. (1) Goldsmith et al. (2012)
    """

    return -tstar/np.log(nu/nl*gl/gu)
    

def tex_cii_thick(tcii, tbg):
    """
    """
    
    return tstar/np.log(1. + tstar/(tcii + J(tbg)))


def dtex_cii_thick(tcii, tbg, dtcii):
    """
    """
    
    fden = 1. + tstar/(tcii + J(tbg))
    den = np.log(fden)
    
    
    return tstar/np.power(den, 2.) * 1./(fden) * (tstar/tcii**2.) * dtcii


def tex_cii(tcii, tau_cii, tbg):
    """
    """
    
    return tstar/np.log(1. + tstar*(1. - np.exp(-tau_cii))/(tcii + J(tbg)))


def dtex_cii(tcii, tau_cii, tbg, dtcii, dtau):
    """
    """
    
    err1 = dtex_cii_dtcii(tcii, tau_cii, tbg, dtcii)
    err2 = dtex_cii_dtau(tcii, tau_cii, tbg, dtau)
    
    return np.sqrt(np.power(err1, 2.) + np.power(err2, 2.))


def dtex_cii_dtau(tcii, tau_cii, tbg, dtau):
    """
    """
    
    return dtex_cii_1(tcii, tau_cii, tbg) * np.exp(-tau_cii)/(tcii + J(tbg)) * dtau


def dtex_cii_dtcii(tcii, tau_cii, tbg, dtcii):
    """
    """
    
    return dtex_cii_1(tcii, tau_cii, tbg) * (1. - np.exp(-tau_cii))/np.power(tcii + J(tbg), 2.) * dtcii
    

def dtex_cii_1(tcii, tau_cii, tbg):
    """
    """
    
    return np.power(tex_cii(tcii, tau_cii, tbg), 2.) * 1./(1 + tstar*(1. - np.exp(-tau_cii))/(tcii + J(tbg)))
