#!/usr/bin/env python

from crrlpy import frec_calc as fc
from crrlpy import crrls
from lmfit import Model
from matplotlib.ticker import MaxNLocator
from crrlpy.models import rrlmod
from astropy.table import Table

import glob
import re
import pylab as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('font', weight='bold')
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def parse_fit_pars(data, mc, fit, n, f0, residuals):
    """
    Converts the fitted line parameters to a Table.
    """
    
    #data = np.zeros((14))
    
    # Store data
    data[0] = n
    data[1] = f0
    data[2] = fit.params['v{0}_center'.format(mc)].value
    data[3] = fit.params['v{0}_center'.format(mc)].stderr
    data[4] = fit.params['v{0}_amplitude'.format(mc)].value*crrls.dv2df(f0*1e6, 1e3)
    data[5] = fit.params['v{0}_amplitude'.format(mc)].stderr*crrls.dv2df(f0*1e6, 1e3)
    dD = 2*fit.params['v{0}_sigma'.format(mc)].value
    dL = 2*fit.params['v{0}_gamma'.format(mc)].value
    dv = crrls.line_width(dD, dL)
    data[6] = dv
    ddD = 2*fit.params['v{0}_sigma'.format(mc)].stderr
    ddL = 2*fit.params['v{0}_gamma'.format(mc)].stderr
    ddv = crrls.line_width_err(dD, dL, ddD, ddL)
    data[7] = ddv
    data[8] = crrls.voigt_peak(fit.params['v{0}_amplitude'.format(mc)].value, 
                               fit.params['v{0}_sigma'.format(mc)].value, 
                               fit.params['v{0}_gamma'.format(mc)].value)
    data[9] = crrls.voigt_peak_err(data[8], 
                                   fit.params['v{0}_amplitude'.format(mc)].value, 
                                   fit.params['v{0}_amplitude'.format(mc)].stderr, 
                                   fit.params['v{0}_sigma'.format(mc)].value, 
                                   fit.params['v{0}_sigma'.format(mc)].stderr)
    data[10] = 2*fit.params['v{0}_sigma'.format(mc)].value
    data[11] = 2*fit.params['v{0}_sigma'.format(mc)].stderr
    data[12] = 2*fit.params['v{0}_gamma'.format(mc)].value
    data[13] = 2*fit.params['v{0}_gamma'.format(mc)].stderr
    data[14] = crrls.get_rms(residuals)

    return data

def save_log(data, log):
    """
    """
    
    table = Table(rows=data, names=('n', 'f0 (MHz)', 'center (km/s)', 'center_err (km/s)',
                                    'itau (Hz)', 'itau_err (Hz)', 'FWHM (km/s)', 'FWHM_err (km/s)',
                                    'tau', 'tau_err', 'FWHM_gauss (km/s)', 'FWHM_gauss_err (km/s)',
                                    'FWHM_lorentz (km/s)', 'FWHM_lorentz_err (km/s)', 'residuals'),
                  dtype=('i3', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    table.write(log, format='ascii.fixed_width')

if __name__ == '__main__':
    
    dD_fix0 = 3.4/2.#crrls.sigma2FWHM(1.4456523-0*0.0267016)/2.
    
    trans = 'alpha'
    frng = 'all'
    stacks = glob.glob('CI{0}_only_n*.ascii'.format(trans))
    crrls.natural_sort(stacks)
    prop = {'n':['812-863', 
                 '760-806', 
                 '713-748',
                 '668-709',
                 '623-665',
                 '580-621'],
            'ns':[37,36,36,36,37,37]}
    
    vel = []
    tau = []
    wei = []
    fit3 = []
    res3 = []
    n = []
    f0 = []
    
    data0 = np.empty((len(stacks), 15))
    
    pdf = PdfPages('C{0}_3c.pdf'.format(trans))
    
    for i,stack in enumerate(stacks):
    
        data = np.loadtxt(stack)
        vel.append(data[:,0])
        tau.append(data[:,1])
        wei.append(data[:,2])
        
        nnow = int(re.findall('\d+', stack)[0])
        n.append(nnow)
        
        dn = fc.set_dn(trans)
        specie, trans, nn, freq = fc.make_line_list('CI', 1500, dn)
        nii = crrls.best_match_indx2(n[i], nn)
        f0.append(freq[nii])
        
        tmin = min(tau[i])
        weight = np.power(wei[i], 2)
        
        v1 = Model(crrls.Voigt, prefix='v0_')
        pars3 = v1.make_params()
        mod3 = v1
        
        pars3['v0_gamma'].set(value=0.1, vary=True, expr='', min=0.0)
        pars3['v0_center'].set(value=-47., vary=True, max=-30, min=-49)
        pars3['v0_amplitude'].set(value=-0.1, vary=True, max=-1e-8)
        pars3['v0_sigma'].set(value=dD_fix0, vary=False)
        
        fit3.append(mod3.fit(tau[i], pars3, x=vel[i], weights=weight))

        # Plot things
        res3.append(tau[i] - fit3[i].best_fit)
        
        voigt0 = crrls.Voigt(vel[i], 
                             fit3[i].params['v0_sigma'].value, 
                             fit3[i].params['v0_gamma'].value, 
                             fit3[i].params['v0_center'].value, 
                             fit3[i].params['v0_amplitude'].value)
        
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1, 1, 1, adjustable='datalim')
        ax.plot(vel[i], tau[i], 'k-', drawstyle='steps', lw=1)
        ax.plot(vel[i], voigt0, 'g-')
        ax.plot(vel[i], fit3[i].best_fit, 'b-', lw=0.5)
        ax.plot(vel[i], res3[i], 'b:', lw=1)
        ax.plot(vel[i], [0]*len(vel[i]), 'k--')
        
        ax.text(0.8, 0.125, r"{0:.2f} MHz".format(f0[i]),
                size="large", transform=ax.transAxes, alpha=1)
        ax.text(0.8, 0.075, r"C$\{0}$({1})".format(trans, nnow),
                size="large", transform=ax.transAxes, alpha=1)
        
        ax.set_xlim(min(vel[i]),max(vel[i]))
        ax.set_ylim(min(tau[i])-max(res3[i]),max(tau[i])+max(res3[i]))
        
        #if (i+1)%2 != 0:
        ax.set_ylabel(r"$\tau_{\nu}$", fontweight='bold', fontsize=20)
        ax.set_xlabel(r"Radio velocity (km s$^{-1}$)", fontweight='bold')
        
        pdf.savefig(fig)
        plt.close(fig)
        
        ## Store data
        parse_fit_pars(data0[i], 0, fit3[i], nnow, f0[i], res3[i])

    pdf.close()
    
    log = 'CI{0}_-47kms_nomod_1c.log'.format(trans)
    save_log(data0, log)