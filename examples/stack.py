#!/usr/bin/env python

import re
import glob
import numpy as np
import pylab as plt
from crrlpy import crrls
from crrlpy import utils
from crrlpy import spec as cs

if __name__ == '__main__':
    
    # Define which things should be stacked
    # These should be spectrum with one axis being the velocity and the other the line optical depth. Aditionally a weight axis is also good.
    specs = sorted(glob.glob('CIalpha_only_n*.ascii'))
    
    # Create a container for the each spectrum.
    lines = np.empty(len(specs), dtype=object)
    qns = np.zeros(len(specs)) # and their principal quantum numbers.
    
    # Loop over files loading the data and putting it into Spectrum objects
    for i,spec in enumerate(specs):
        qns[i] = re.findall('\d+', spec)[0] # Find the n of the line, will not work unless the filenames have it in there.
        data = np.loadtxt(spec) # Load the data
        w = data[:,2] # Set the weights
        #w = np.ones(data[:,2].shape) # I was testing this
        lines[i] = cs.Spectrum(data[:,0], data[:,1], w, spw=qns[i]) # Make a Spectrum object
    
    # Prepare to stack.
    line_name = 'CRRL_{0:.0f}'.format(round(np.mean(qns))) # Give it a name, completely irrelevant.
    stack = cs.Stack(line_name, qns, lines, vmin=-250, vmax=250) # Stack the spectrum objects. Select a velocity range.
    rms = stack.stack_interpol(rms=True, ch0=0, chf=29) # Stack by interpolating to a common velocity axis. The arguments are not necessary.
    stack.save('CRRL_LBA_MID_stack_n{0:.0f}.ascii'.format(round(np.mean(qns)))) # Save it, or not.
    
    # Plot...
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(1, 1, 1, adjustable='datalim')
    ax.step(stack.x[v0:vf], stack.y[v0:vf], 'k-', drawstyle='steps', lw=1, where='pre')
    ax.set_xlabel('Velocity (km s$^{-1}$)')
    ax.set_ylabel('"$T_{A}$ (K)"')
    plt.savefig('CRRL_stack_LBA_MID.pdf', 
                bbox_inches='tight', 
                pad_inches=0.06)
    plt.close()
    
    # Another plot...
    x = np.arange(1, len(qns)+1)
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(1, 1, 1, adjustable='datalim')
    ax.plot(x, rms, 'bo', ls='none')
    ax.plot(x, rms[0]/np.sqrt(x), 'b-')
    ax.set_xlabel('Number of stacked spectra')
    ax.set_ylabel('rms')
    plt.savefig('CRRL_stack_rms.pdf', 
                bbox_inches='tight', 
                pad_inches=0.06)
    plt.close()