#!/usr/bin/env python
 
from astropy.io import fits
from crrlpy.crrls import natural_sort

import logging
import sys
import re
import os
import argparse
import numpy as np

blankval = np.nan

def get_cube_dims(fitslist, chan_id='chan'):
    hdulist = fits.open(fitslist[0])
    head = hdulist[0].header
    nx = head['NAXIS1']
    ny = head['NAXIS2']
    #nv = len(fitslist)
    regex = r'{0}(.+?)(\.|\_)'.format(chan_id)
    ch0 = int(re.search(r'{0}(.+?)(\.|\_|\-)'.format(chan_id), fitslist[0]).group(1))
    print "file {0} is channel {1}".format(fitslist[0], ch0)
    chf = int(re.search(r'{0}(.+?)(\.|\_|\-)'.format(chan_id), fitslist[-1]).group(1))
    print "file {0} is channel {1}".format(fitslist[-1], chf)
    nv = chf - ch0
    
    return nx, ny, nv, ch0

def main(outfits, fitslist, stokeslast=True, chan_id='chan', chan_end='.', nzeros=4, clobber=False):
    
    # Get cube dimensions from first image
    nx, ny, nv, ch0 = get_cube_dims(fitslist, chan_id)
    print "Starting channel: {0}".format(ch0)
    print "Number of channels: {0}".format(nv)
    # Create the cube, only stokes I
    cube = np.ma.empty([1, nv, ny, nx], dtype=float)
    
    # Loop over images copying their data to the cube
    for i in range(nv):
        
        print '{0}{1}.'.format(chan_id, str(i+ch0).zfill(nzeros))
        fitsch = filter(lambda x: '{0}{1}{2}'.format(chan_id, str(i+ch0).zfill(nzeros), chan_end) in x, fitslist)
        
        if fitsch:
            print fitsch
            hdulist = fits.open(fitsch[0])
            data = hdulist[0].data
            
            if len(data.shape) > 3:
                data = data[0][0]
            if len(data.shape) > 2:
                data = data[0]
        else:
            print "Inserting blank channel"
            data = np.ones([ny, nx])*blankval
            
        cube[0,i,:,:] = np.ma.masked_invalid(data)
    
    cube.fill_value = np.nan
    
    hdulist = fits.PrimaryHDU(cube.filled())
    # Copy the header from the first channel image
    hdulist.header = fits.open(fitslist[0])[0].header.copy()
    
    if not stokeslast:
        # Change 3rd axis for 4th
        # In case stokeslast = False
        ctype3 = hdulist.header['CTYPE3']
        crval3 = hdulist.header['CRVAL3']
        cdelt3 = hdulist.header['CDELT3']
        crpix3 = hdulist.header['CRPIX3']
        cunit3 = hdulist.header['CUNIT3']
        hdulist.header['CTYPE3'] = hdulist.header['CTYPE4']
        hdulist.header['CRVAL3'] = hdulist.header['CRVAL4']
        hdulist.header['CDELT3'] = hdulist.header['CDELT4']
        hdulist.header['CRPIX3'] = hdulist.header['CRPIX4']
        hdulist.header['CUNIT3'] = hdulist.header['CUNIT4']
        hdulist.header['CTYPE4'] = ctype3
        hdulist.header['CRVAL4'] = crval3
        hdulist.header['CDELT4'] = cdelt3
        hdulist.header['CRPIX4'] = crpix3
        hdulist.header['CUNIT4'] = cunit3
    
    #hdulist.header.append(('BLANK', blankval))
    # Write to a fits file
    hdulist.writeto(outfits, clobber=clobber)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('fitslist', help="List of fits files to combine into a cube.", 
                        nargs=argparse.REMAINDER)
    parser.add_argument('-o', '--outfits', 
                        help="Output cube name.", required=True)
    parser.add_argument('-s', '--stokeslast', 
                        help="Is the Stokes axis the last on the fits files?",
                        action='store_true', dest='stokeslast')
    parser.add_argument('-c', '--chan_id', 
                        help="String before the channel number. (string, Default: chan)",
                        type=str, default='chan')
    parser.add_argument('--chan_end', 
                        help="String after the channel number. (string, Default: .)",
                        type=str, default='.')
    parser.add_argument('-n', '--nzeros', 
                        help="Number of zeros to the right of the channel number. (int, Default: 4)",
                        type=int, default=4)
    parser.add_argument('--clobber', 
                        help="Overwrite existing fits files?",
                        action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose output?")
    parser.add_argument('-l', '--logfile', type=str, default=None,
                        help="Where to store the logs.\n" \
                             "(string, Default: output to console)")
    args = parser.parse_args()
    
    if args.verbose:
        loglev = logging.DEBUG
    else:
        loglev = logging.ERROR
    
    # Prepare the logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=args.logfile, level=loglev, format=formatter)
    
    logger = logging.getLogger(__name__) 
    
    outfits = args.outfits
    fitslist = args.fitslist
    
    natural_sort(fitslist)
    
    logger.debug(fitslist)

    main(outfits, fitslist, args.stokeslast, args.chan_id, args.chan_end, args.nzeros, args.clobber)
