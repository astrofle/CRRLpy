#!/usr/bin/env python

import argparse

import numpy as np

from scipy.constants import c, m_e, physical_constants
from astropy import units as u

def line_freq(Z, R_X, n, dn):
    """
    Uses Rydberg formula to get the frequency
    of a transition to quantum number n.
    """
    
    return (Z**2)*R_X*c*((1./(n**2))-(1./((n + dn)**2)))

def set_specie(specie):
    """
    """
    
    # data for species (table 1 RG92)
    # [atomic.mass, ion.potential, abundance, V_X-V_H, Z]
    if 'HI' in specie:
        X = [1.0078,13.6,1.0,0.0,1.0]
        name = 'HI'
    if 'HeI' in specie:
        X = [4.0026,24.6,0.1,122.1,1.0]
        name = 'HeI'
    if 'CI' in specie:
        #X = [12.0000,11.4,3.e-4,149.5,6.0]
        X = [12.0000,11.4,3.e-4,149.5,1.0]
        name = 'CI'
    if 'NI' in specie:
        X = [14.0067,1,1,1,1.0]
        name = 'NI'
    if 'SI' in specie:
        #X = [37.9721,10.3,2.e-5,158.0,16.0]
        X = [37.9721,10.3,2.e-5,158.0,1.0]
        name = 'SI'
    # isotopes
    if 'CI13' in specie:
        X = [13.00335,-1.0,-1.0,-1.0,1.0]
        name = 'CI13'
    if 'CI14' in specie:
        X = [14.003241,-1.0,-1.0,-1.0,1.0]
        name = 'CI14'
        
    return X

def set_trans(dn):
    """
    Sets a name depending on the difference between
    atomic levels.
    :param dn: Separation between ni and nf, :math:`dn=ni-nf`.
    :returns: alpha, beta, gamma, delta or epsilon depending on :paramref:`dn`.
    """
    if dn == 1:
        name = 'alpha'
    if dn == 2:
        name = 'beta'
    if dn == 3:
        name = 'gamma'
    if dn == 4:
        name = 'delta'
    if dn == 5:
        name = 'epsilon'
        
    return name

def set_dn(name):
    """
    Sets the value of Delta n depending on 
    the transition name.
    """
    
    if 'alpha' in name:
        dn = 1
    if 'beta' in name:
        dn = 2
    if 'gamma' in name:
        dn = 3
    if 'delta' in name:
        dn = 4
    if 'epsilon' in name:
        dn = 5
        
    return dn

def make_line_list(line, n_min=1, n_max=1500, unitless=True):
    """
    Creates a list of frequencies for the
    corresponding n level. The frequencies
    are in MHz.
    """
    
    n = np.arange(n_min, n_max)
    
    # Define the electron mass in atomic mass units
    m_e_amu = m_e/physical_constants['atomic mass constant'][0]
    
    # set the specie
    X = set_specie(line)
    dn = set_dn(line)
    
    M_X = X[0]
    R_X = 10.97373/(1.0 + (m_e_amu/M_X))
    Z = X[4]
    
    freq = line_freq(Z, R_X, n, dn)
    
    if not unitless:
        freq = freq*u.MHz
    
    return line, n, freq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--n_min', type=int,
                        dest='n_min', default=1, help="Minimum n number")
    parser.add_argument('-n', '--n_max', type=int,
                        dest='n_max', default=1500, help="Maximum n number")
    parser.add_argument('-l', '--line', dest='line', default='CI', type=str,
                        help="Line name. E.g., CIalpha, HeIbeta, HIalpha, CI13alpha, CI14gamma or SIepsilon")
    args = parser.parse_args()
    
    n_min = args.n_min
    n_max = args.n_max
    line = args.line
    
    specie, trans, n, freq = make_line_list(line, n_min, n_max)
    
    # Write the line list to a file
    out = 'RRL_{0}{1}.txt'.format(specie, trans)
    with open(out, 'w') as outf:
        outf.write('#SPECIES-NAME,  TRANSITION-TYPE,  N-LEVEL,  FREQUENCY-[MHZ]\n')
        for i, ni in enumerate(n):
            outf.write('{0}  {1}  {2}  {3}\n'.format(specie, trans, ni, freq[i]))

if __name__ == '__main__':
    main()