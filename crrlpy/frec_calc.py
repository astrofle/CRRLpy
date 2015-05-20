#!/usr/bin/env python

import argparse
import numpy as np
from scipy.constants import c, m_e, physical_constants

def line_freq(Z, R_X, n, dn):
    """
    Uses Rydberg formula to get the frequency
    of a transition to quantum number n.
    """
    return (Z**2)*R_X*c*((1./(n**2))-(1./((n + dn)**2)))

def set_specie(specie):
    # data for species (table 1 RG92)
    # [atomic.mass, ion.potential, abundance, V_X-V_H, Z]
    if specie == 'HI':
        X = [1.0078,13.6,1.0,0.0,1.0]
        name = 'HI'
    if specie == 'HeI':
        X = [4.0026,24.6,0.1,122.1,1.0]
        name = 'HeI'
    if specie == 'CI':
        #X = [12.0000,11.4,3.e-4,149.5,6.0]
        X = [12.0000,11.4,3.e-4,149.5,1.0]
        name = 'CI'
    if specie == 'NI':
        X = [14.0067,1,1,1,1.0]
        name = 'NI'
    if specie == 'SI':
        #X = [37.9721,10.3,2.e-5,158.0,16.0]
        X = [37.9721,10.3,2.e-5,158.0,1.0]
        name = 'SI'
    # isotopes
    if specie == 'CI13':
        X = [13.00335,-1.0,-1.0,-1.0,1.0]
        name = 'CI13'
    if specie == 'CI14':
        X = [14.003241,-1.0,-1.0,-1.0,1.0]
        name = 'CI14'
        
    return X

def set_trans(dn):
    """
    Sets a name depending on the difference between
    atomic levels.
    """
    if dn == 1:
        name = 'alpha'
    if dn == 2:
        name = 'beta'
    if dn == 3:
        name = 'gamma'
    if dn == 4:
        name = 'delta'
    return name

def set_dn(name):
    """
    Sets the upper level depending on 
    the transition name.
    """
    if name == 'alpha':
        dn = 1
    if name == 'beta':
        dn = 2
    if name == 'gamma':
        dn = 3
    if name == 'delta':
        dn = 4
    return dn

def make_line_list(specie, nmax, dn):
    n = np.arange(1, nmax)
    
    # Define the electron mass in atomic mass units
    m_e_amu = m_e/physical_constants['atomic mass constant'][0]
    
    # set the specie
    X = set_specie(specie)
    # Set the transition name
    trans = set_trans(dn)
    
    M_X = X[0]
    R_X = 10.97373/(1.0 + (m_e_amu/M_X))
    Z = X[4]
    
    freq = line_freq(Z, R_X, n, dn)
    
    return specie, trans, n, freq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nmax', type=int,
                        dest='nmax', default=1, help="Maximum n number")
    parser.add_argument('-s', '--specie', dest='specie', default='CI', type=str,
                        help="Specie name. Can be CI, HeI, HI, CI13, CI14 or SI")
    parser.add_argument('-d', '--dn', dest='dn', default=1, type=int,
                        help="Difference between quantum levels. Can be up to 4.")
    args = parser.parse_args()
    
    nmax = args.nmax
    specie = args.specie
    dn = args.dn
    
    specie, trans, n, freq = make_line_list(specie, nmax, dn)
    
    # Write the line list to a file
    out = 'RRL_{0}{1}.txt'.format(specie, trans)
    with open(out, 'w') as outf:
        outf.write('#SPECIES-NAME,  TRANSITION-TYPE,  N-LEVEL,  FREQUENCY-[MHZ]\n')
        for i, ni in enumerate(n):
            outf.write('{0}  {1}  {2}  {3}\n'.format(specie, trans, ni, freq[i]))

if __name__ == '__main__':
    main()