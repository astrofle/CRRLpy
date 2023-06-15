#!/usr/bin/env python

__docformat__ = 'reStructuredText'

import argparse

import numpy as np

from scipy.constants import c, m_e, physical_constants
from astropy import units as u


def line_freq(Z, R_X, n, dn):
    """
    Uses the Rydberg formula to get the frequency
    of a transition to quantum number n for a given atom.
    
    :param Z: Charge of the atom.
    :type Z: int
    :param R_X:
    :type R_X: float
    :param n: Principal quantum number of the transition. :math:`n+\\Delta n\\rightarrow n`.
    :type n: int
    :param dn: Difference between the principal quantum number of the initial state \
    and the final state. :math:`\\Delta n=n_{f}-n_{i}`.
    :type dn: int
    :returns: The frequency of the transition in MHz.
    :rtype: float
    """
    
    return (Z**2)*R_X*c*((1./(n**2))-(1./((n + dn)**2)))


def set_specie(specie):
    """
    Sets atomic constants based on the atomic specie.
    
    :param specie: Atomic specie.
    :type specie: string
    :returns: Array with the atomic mass in a.m.u., ionization potential, abundance relative to HI, :math:`V_{X}-V_{H}` and the electric charge.
    
    :Example:
    
    >>> set_specie('CI')
    [12.0, 11.4, 0.0003, 149.5, 1.0]

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
    Sets a name depending on the difference between atomic levels.
    
    :param dn: Separation between :math:`n_{i}` and :math:`n_{f}`, :math:`\\Delta n=n_{i}-n_{f}`.
    :type dn: int
    :returns: alpha, beta, gamma, delta or epsilon depending on :math:`\\Delta n`.
    :rtype: string
    
    :Example:
    
    >>> set_trans(5)
    'epsilon'
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
    if dn == 6:
        name = 'zeta'
    if dn == 7:
        name = 'eta'
        
    return name


def set_dn(name):
    """
    Sets the value of Delta n depending on the transition name.
    
    :param name: Name of the transition.
    :type name: string
    :returns: :math:`\\Delta n` for the given transition.
    :rtype: int
    
    :Example:
    
    >>> set_dn('CIalpha')
    1
    >>> set_dn('CIdelta')
    4
    """
    
    if 'alpha' in name:
        dn = 1
    elif 'beta' in name:
        dn = 2
    elif 'gamma' in name:
        dn = 3
    elif 'delta' in name:
        dn = 4
    elif 'epsilon' in name:
        dn = 5
    elif 'zeta' in name:
        dn = 6
    elif 'eta' in name:
        dn = 7
        
    return dn


def make_line_list(line, n_min=1, n_max=1500, unitless=True):
    """
    Creates a list of frequencies for the corresponding line. The frequencies are in MHz.
    
    :param line: Line to compute the frequencies for.
    :type line: string
    :param n_min: Minimum n number to include in the list.
    :type n_min: int
    :param n_max: Maximum n number to include in the list.
    :type n_max: int
    :param unitless: If True the list will have no units. If not the list will be of astropy.units.Quantity_ objects.
    :type unitless: bool
    :returns: 3 lists with the line name, principal quantum number and frequency of the transitions.
    :rtype: list
    
    .. _astropy.units.Quantity: http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html#astropy.units.Quantity
    """
    
    n = np.arange(n_min, n_max)
    
    # Define the electron mass in atomic mass units
    m_e_amu = m_e/physical_constants['atomic mass constant'][0]
    
    # set the specie
    X = set_specie(line)
    dn = set_dn(line)
    trans = set_trans(dn)
    
    M_X = X[0]
    R_X = 10.97373/(1.0 + (m_e_amu/M_X))
    Z = X[4]
    
    freq = line_freq(Z, R_X, n, dn)
    
    if not unitless:
        freq = freq*u.MHz
    
    return line, n, freq, trans


def main():
    """
    Main body of the program. Useful for calling as a script.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--n_min', type=int,
                        dest='n_min', default=1, help="Minimum n number")
    parser.add_argument('-n', '--n_max', type=int,
                        dest='n_max', default=10000, help="Maximum n number")
    parser.add_argument('-l', '--line', dest='line', default='CI', type=str,
                        help="Line name. E.g., CIalpha, HeIbeta, HIalpha, CI13alpha, CI14gamma or SIepsilon")
    args = parser.parse_args()
    
    n_min = args.n_min
    n_max = args.n_max
    line = args.line
    
    line, n, freq, trans = make_line_list(line, n_min, n_max)
    
    specie = line[:line.index(trans)]
    
    # Write the line list to a file
    out = 'RRL_{0}{1}.txt'.format(specie, trans)
    with open(out, 'w') as outf:
        outf.write('#SPECIES-NAME,  TRANSITION-TYPE,  N-LEVEL,  FREQUENCY-[MHZ]\n')
        for i, ni in enumerate(n):
            outf.write('{0}  {1}  {2}  {3}\n'.format(specie, trans, ni, freq[i]))

if __name__ == '__main__':
    main()
