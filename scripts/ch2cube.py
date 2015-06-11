#!/usr/bin/env python 
from astropy.io import fits
from crrlpy.crrls import natural_sort
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
    ch0 = int(re.search(r'{0}(.+?)\.'.format(chan_id), fitslist[0]).group(1))
    #print "file {0} is channel {1}".format(fitslist[0], ch0)
    chf = int(re.search(r'{0}(.+?)\.'.format(chan_id), fitslist[-1]).group(1))
    #print "file {0} is channel {1}".format(fitslist[-1], chf)
    nv = chf - ch0
    
    return nx, ny, nv, ch0

def make_fits(data):
    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])
    return hdulist

def update_header(header, fitslist):
    hdulist = fits.open(fitslist[0])
    head = hdulist[0].header
    # copy the header from the first fits in the list
    header = head.copy()
    #for key in head.keys():
        ##if key != 'CRVAL4':
        #if key:
            #print "{0}:  {1}".format(key, head[key])
            #header.append((key, head[key]))
        
            
    #header.append('BSCALE', head['BSCALE'])
    #header.append('BZERO', head['BZERO'])
    #header.append('BMAJ', head['BMAJ'])
    #header.append('BMIN', head['BMIN'])
    #header.append('BPA', head['BPA'])
    #header.append('BTYPE', head['BTYPE'])
    #header.append('BUNIT', head['BUNIT'])
    #header.append('EQUINOX', head['EQUINOX'])
    #header.append('EQUINOX', head['LONPOLE'])

def main(outfits, fitslist, stokeslast=True, chan_id='chan'):
    
    # Get cube dimensions from first image
    nx, ny, nv, ch0 = get_cube_dims(fitslist, chan_id)
    print nv
    # Create the cube, only stokes I
    cube = np.empty([1, nv, ny, nx], dtype=float)
    
    # Get the first channel number
    #ch0 = int(re.search('chan(.+?).', fitslist[0]))
    #pch = ch0
    
    for i in range(nv):
        fitsch = filter(lambda x: '{0}{1:d}.'.format(chan_id, i+ch0) in x, fitslist)
        if fitsch:
            print fitsch
            hdulist = fits.open(fitsch[0])
            data = hdulist[0].data[0][0]
        else:
            print "Inserting blank channel"
            data = np.ones([ny, nx])*blankval
        cube[0,i,:,:] = data
    
    # Loop over images copying their data to the cube
    #for i,image in enumerate(fitslist):
        #ch = int(re.search('chan(.+?).', fitslist[0]))
        #if ch - pch != 1:
            #for j in range(ch - pch):
                #cube[0,i+j,:,:] = data
        #hdulist = fits.open(image)
        ##head = hdulist[0].header
        ## Data is stokes I and one velocity channel
        #data = hdulist[0].data[0][0] 
        #cube[0,i,:,:] = data
        #pch = ch
        
    hdulist = fits.PrimaryHDU(cube)
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
    hdulist.writeto(outfits)
    
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
    args = parser.parse_args()
    
    outfits = args.outfits
    fitslist = args.fitslist
    
    natural_sort(fitslist)

    main(outfits, fitslist, args.stokeslast, args.chan_id)
