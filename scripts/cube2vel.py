"""

"""

import sys
import glob
import argparse
from crrlpy import crrls
from astropy.io import fits


def add_axis(head, axis, naxis, crpix, crval, cdelt, ctype, cunit):
    """
    Adds an axis to the fits cube
    """
    
    head['NAXIS'] = head['NAXIS'] + 1
    head['NAXIS{0}'.format(axis)] = naxis
    head['CRPIX{0}'.format(axis)] = crpix
    head['CRVAL{0}'.format(axis)] = crval
    head['CDELT{0}'.format(axis)] = cdelt
    head['CTYPE{0}'.format(axis)] = ctype
    head['CUNIT{0}'.format(axis)] = cunit
   

def find_line_qn_cube(freq, transition, z, qnidx=0):
    """
    """

    qns, reff = crrls.find_lines_sb(freq, transition, z)

    if not reff:
        print('No {0} line found.'.format(transition))
        print('Will now exit.')
        sys.exit(0)

    # Handle multiple lines in one cube.
    if len(reff) > 1:
        print('More than one line in the cube.')
        if qnidx is None:
            print('Split the cube or set `qnidx`.')
            print('Will now exit.')
            sys.exit(0)
    qns = qns[qnidx]
    reff = reff[qnidx]

    return qns, reff


def cube2vel(cube, transition='RRL_CIalpha', z=0, f_col=3, v_col=3, unit='m/s', qnidx=0, n=None):
    """
    Read the frequency axis of a fits cube and creates and axis 
    with velocity with respect to the specified transition.
    
    """
    
    hdu = fits.open(cube, mode='update')
    head = hdu[0].header
    freq = crrls.get_axis(head, f_col)
    #print(freq.mean(), freq.ptp())
    
    # Invert frequency axis when searching for lines if necessary
    fi = 1
    if freq[0] > freq[-1]: fi = -1

    ## Find the lines in the cube
    #qns, reff = crrls.find_lines_sb(freq[::fi]*1e-6, transition, z) # returns in MHz

    #if not reff:
    #    print('No {0} line found.'.format(transition))
    #    print('Will now exit.')
    #    sys.exit(0)
    #
    ## Handle multiple lines in one cube.
    #if len(reff) > 1:
    #    print('More than one line in the cube.')
    #    if qnidx is None:
    #        print('Split the cube or set `qnidx`.')
    #        print('Will now exit.')
    #        sys.exit(0)
    #qns = qns[qnidx]
    #reff = reff[qnidx]

    if n is None:
        qns, reff = find_line_qn_cube(freq[::fi]*1e-6, transition, z, qnidx=0)
    else:
        print("Principal quantum number given.")
        print("Will use the rest frequency for that transition.")
        qns = n
        reff = crrls.n2f(n, transition)

    # Get a velocity axis
    vel = crrls.freq2vel(reff*1e6, freq)
    
    if unit == 'km/s':
        vel = vel*1e-3

    add_axis(head, v_col, len(vel), 1, vel[0], vel[1]-vel[0], 'VELO', unit)
    # Add principal quantum number to header.
    head['PQN'] = (qns, 'Principal quantum number')
    
    hdu[0].header = head
    
    hdu.flush()
    hdu.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('cube', type=str,
                        help="Cube to process.\n" \
                             "E.g., \"lba_hgh_*.fits\" (string).\n" \
                             "Wildcards and [] accepted.")
    parser.add_argument('-t', '--transition', type=str, default='RRL_CIalpha',
                        help="Transition to convert in the spectra.\n" \
                             "E.g., RRL_CI13beta" \
                             "Default: RRL_CIalpha")
    parser.add_argument('--z', type=float, default=0.0, dest='z',
                        help="Redshift to apply to the transition rest frequency.\n" \
                             "Default: 0")
    parser.add_argument('--f_col', type=int, default=3,
                        help="Header axis with frequency values.\n" \
                             "This will be converted to velocity. Default: 3")
    parser.add_argument('--v_col', type=int, default=3,
                        help="Header axis where the velocity information will be stored.\n" \
                             "Default: 3")
    parser.add_argument('-n', '--principal_qn', type=int)
    args = parser.parse_args()
    
    cube2vel(args.cube, args.transition, args.z, args.f_col, args.v_col, n=args.principal_qn)
