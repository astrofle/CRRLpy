#!/usr/bin/env python

#from __future__ import division

import numpy as np
import crrls
import utils
from scipy import interpolate

class Spectrum(object):
    """
    
    Construction::
    
    spec = Spectrum(x, y, w=None, spw=None, stack=None)
    
    Parameters
    ----------
    x : array_like
        | Spectrum x axis.
        | If x is a frequency axis, then its units should be MHz.
    y : array_like
        Spectrum y axis.
    w : array_like, optional
        Spectrum weight axis
    spw : :obj:`int`
        Spectrum spectral window.
    """
    
    def __init__(self, x, y, w=[], spw=None, stack=None):
        
        # User provided values
        self.x = np.ma.masked_invalid(x)
        self.y = np.ma.masked_invalid(y)
        self.sy = np.ma.masked_invalid(y) # For smoothing purposes
        if w != []:
            self.z = np.ma.masked_invalid(w)
        else:
            self.z = np.ma.masked_invalid(np.ones(len(x)))
        self.z.fill_value = 1.
        self.spw = spw
        self.stack = stack
        
        # Determine the global mask from x, y and z
        self.mask = self.x.mask | self.y.mask | self.z.mask
        # Apply the global mask to the data
        self.x.mask = self.mask
        self.y.mask = self.mask
        self.z.mask = self.mask
        self.sy.mask = self.mask
        
        self.good_lines = {}
        self.lines = {}
        self.bw = self.x.compressed().max() - self.x.compressed().min()
        self.dx = utils.get_min_sep(self.x.compressed())
        self.nx = len(self.x.data)
    
    def apply_bandpass_corr(self, bandpass_x, bandpass_y, offset=1000., overwrite=False):
        """
        Applies a bandpass correction to the y axis of Spectrum.
        
        Parameters
        ----------
        bandpass : :obj:`numpy.ma.array`
                   Bandpass to apply to Spectrum.
        offset : :obj:`float`, optional
              | Offset to apply to the y axis data. Used to avoid division by 0.
              | Should be the same value used when deriving the bandpass solution.
        overwrite : :obj:`bool`, optional
                    Should the bandpass applied overwrite the Spectrum.bp?
        """
        
        bp = np.ma.masked_invalid(np.interp(self.x.data, bandpass_x, bandpass_y))
        bp.fill_value = offset
            
        self.y_bpcorr = np.ma.masked_invalid(((self.y.data + offset)/bp.filled() - 1.)*offset)
        
        if overwrite:
            self.bp = bp
    
    def bandpass_corr(self, order, offset=1000.):
        """
        Fits a polynomial to the unmasked elements of spec
        and uses it to correct the bandpass.
        
        Parameters
        ----------
        order : :obj:`int`
                Order of the polynomial to be fit.
        offset : :obj:`float`, optional
               | Offset to apply to the y axis data. 
               | Used to avoid division by 0.
        
        """
        
        # Add offset to avoid zeros
        oy = self.y.compressed() + offset
        
        # Use a polynomial to remove the baseline
        bp = np.polynomial.polynomial.polyfit(self.x.compressed(), oy, order, w=self.z.compressed())
        # Interpolate and extrapolate to the original x axis
        b = np.polynomial.polynomial.polyval(self.x.data, bp)
        
        # Flag NaN values in the baseline
        mb = np.ma.masked_invalid(b)
        self.bp = np.ma.masked_equal(mb, 0)
        self.bp.fill_value = offset
        
        self.y_bpcorr = np.ma.masked_invalid(((self.y.data + offset)/self.bp.filled() - 1.)*offset)
    
    def find_lines(self, line, z=0, verbose=False):
        """
        Finds if there are any lines of a given type in the frequency range.
        The line frequencies are corrected for redshift.
        
        Parameters
        ----------
        line : :obj:`string`
               Line type to search for.
        z :    :obj:`float`
            Redshift to apply to the rest frequencies.
        verbose : :obj:`bool`
                  Verbose output?
        
        Returns
        -------
        n : :obj:`numpy.array`
            Principal quantum numbers. 
        reference_frequencies : :obj:`numpy.array`
                                Reference frequencies of the lines inside the spectrum in MHz. 
                                The frequencies are redshift corrected.
        
        See Also
        --------
        crrlpy.crrls.load_ref : Describes the format of line and the available ones.
        
        Examples
        --------
        >>> from crrlpy.spec import Spectrum
        >>> freq = [10, 11]
        >>> temp = [1, 1]
        >>> spec = Spectrum(freq, temp)
        >>> ns, rf = spec.find_lines('RRL_CIalpha')
        >>> ns
        array([ 843.,  844.,  845.,  846.,  847.,  848.,  849.,  850.,  851.,
                852.,  853.,  854.,  855.,  856.,  857.,  858.,  859.,  860.,
                861.,  862.,  863.,  864.,  865.,  866.,  867.,  868.,  869.])
        """
        
        if not isinstance(line, str):
            raise ValueError('line should be a string')
            
        # Load the reference frequencies.
        qn, restfreq = crrls.load_ref(line)
        
        # Correct rest frequencies for redshift.
        reffreq = restfreq/(1.0 + z)
        
        # Check which lines lie within the sub band.
        mask_ref = (self.x.compressed()[0] < reffreq) & \
                   (self.x.compressed()[-1] > reffreq)
        reffreqs = reffreq[mask_ref]
        refqns = qn[mask_ref]
        
        if not line in self.lines.keys():
            
            try:
                self.lines[line].append(refqns)
                self.lines[line+'_freq'].append(reffreqs)
            except KeyError:
                self.lines[line] = [refqns]
                self.lines[line+'_freq'] = [reffreqs]

            self.lines[line] = utils.flatten_list(self.lines[line])
            self.lines[line+'_freq'] = utils.flatten_list(self.lines[line+'_freq'])
            
        nlin = len(reffreqs)
        if verbose:
            print("Found {0} {1} lines within the subband.".format(nlin, line))
            if nlin > 1:
                print("Corresponding to n values: {0}--{1}".format(refqns[0], refqns[-1]))
            elif nlin == 1:
                print("Corresponding to n value {0} and frequency {1} MHz".format(refqns[0], reffreqs[0]))

        return refqns, reffreqs
    
    def find_good_lines(self, line, lines, z=0, separation=0, redge=0.05, ledge=None):
        """
        Find any good lines in the Spectrum.
        
        Parameters
        ----------
        line : :obj:`str`
               Search for good lines of this kind.
        lines : :obj:`str`
               Compare against this kind of lines.
        z : :obj:`float`
               Redshift correction to apply to the rest frequencies.
        separation : :obj:`float`
                Minimum separation between lines to be considered good.
        redge : :obj:`float`
                The line frequency should be this far 
        """
        
        # If no value is given for the left edge, use the same as for the right edge
        if ledge == None:
            ledge = redge
        
        # Find the lines within the Spectrum corresponding to the desired line
        ns, rf = self.find_lines(line, z)
        
        # Find other lines in the Spectrum
        ofs = []
        for l in lines:
            n, f = self.find_lines(l, z)
            ofs.append(list(f))
        
        fofs = utils.flatten_list(ofs)
        
        # Loop over lines checking that their separation from the other lines
        # is larger than separation.
        for i,f in enumerate(rf):
            diff = [abs(of - f) if of != f else separation+1 for of in fofs]
            if all(d > separation for d in diff) and \
                f >= self.x.compressed().min() + self.bw*ledge and \
                f <= self.x.compressed().max() - self.bw*redge:
                try:
                    self.good_lines[line].append(ns[i])
                    self.good_lines[line+'_freq'].append(rf[i])
                except KeyError:
                    self.good_lines[line] = [ns[i]]
                    self.good_lines[line+'_freq'] = [rf[i]]
        try:
            self.good_lines[line]
            self.good_lines[line+'_freq']
        except KeyError:
            self.good_lines[line] = []
            self.good_lines[line+'_freq'] = []
            
    def make_line_mask(self, line, z=0, df=5):
        """
        Creates a list of indexes to mask lines in the spectrum.
        
        Parameters
        ----------
        line : :obj:`str`
               Line to mask.
        
        See Also
        --------
        crrlpy.crrls.load_ref : Describes the format of line and the available ones.
        """
        
        ns, rf = self.find_lines(line, z=z)
        nlines = len(ns)
        rngs = np.zeros((nlines, 2), dtype=int)
        
        for i,f in enumerate(rf):
            rngs[i][0] = utils.best_match_indx(f-df/2., self.x.compressed())
            rngs[i][1] = utils.best_match_indx(f+df/2., self.x.compressed())
            
        return rngs
    
    def mask_edges(self, redge, ledge=None):
        """
        Mask the edges of the Spectrum frequency and amplitude.
        """
        
        if ledge == None:
            ledge = redge
            
        if isinstance(redge, int) and isinstance(ledge, int):
            self.x.mask[:ledge] = True
            self.x.mask[-redge:] = True
            self.y.mask[:ledge] = True
            self.y.mask[-redge:] = True
            self.z.mask[:ledge] = True
            self.z.mask[-redge:] = True
        else:
            raise ValueError('redge and ledge should be integers')
    
    def mask_ranges(self, ranges):
        """
        Masks the Spectrum inside the given ranges.
        
        Parameters
        ----------
        ranges : list of tuples
                 Indexes defining the ranges to be masked.
        
        Examples
        --------
        >>> x = np.arange(0,10)
        >>> y = np.arange(0,10)
        >>> spec = Spectrum(x,y)
        >>> spec.y.compressed()
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> rngs = [[0,2], [7,9]]
        >>> spec.mask_ranges(rngs)
        >>> spec.y.compressed()
        array([2, 3, 4, 5, 6, 9])
        """
        
        for rng in ranges:
            self.x.mask[rng[0]:rng[1]] = True
            self.y.mask[rng[0]:rng[1]] = True
            self.z.mask[rng[0]:rng[1]] = True
            
    def remove_model(self, line, model, z=0, is_freq=False):
        """
        Subtracts the model from the y axis of the Spectrum.
        
        Parameters
        ----------
        line : :obj:`str`
               Line which the model represents.
        model : array_like
               Array with the model to be removed. Should have shape (2xN).
        is_freq : :obj:`bool`
               Is the model x axis in frequency units?
        """
        
        modx = model[0]
        mody = model[1]
        p = np.argsort(modx)
        
        qns, freqs = self.find_lines(line, z=z)
        
        y_mod = np.zeros(len(self.y))
        
        if is_freq:
            # Interpolate the model axis to the spectrum grid
            imody = interpolate.interp1d(modx[p], mody[p],
                                         kind='linear',
                                         bounds_error=False,
                                         fill_value=0.0)
            y_mod += imody(self.x.data)
            
        else:
            for i,n in enumerate(qns):
                # Convert the model velocity axis to frequency
                fm = crrls.vel2freq(freqs[i], modx*1e3)
                p = fm.argsort()
                
                # Interpolate the model axis to the spectrum grid
                imody = interpolate.interp1d(fm[p], mody[p],
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value=0.0)
                y_mod += imody(self.x.data)
        
        # Remove the model
        self.y_minus_mod = self.y - y_mod
        
        # Store the model
        self.model = y_mod
    
    def save(self, filename):
        """
        Saves the spectrum.
        
        Parameters
        ----------
        filename : :obj:`str`
                  Output filename.
        """
        
        np.savetxt(filename, np.c_[self.x.data, self.y.data, self.z.data])
    
    def split_lines(self, reffreqs):
        """
        Splits the spectrum to separate lines.
        
        Parameters
        ----------
        lines : array_like
               List of lines to split.
        """

        nlines = len(reffreqs)
        
        if nlines == 0:
            return
        
        lbw = self.nx/nlines
        
        splits = [[[] for i in range(3)] for j in range(nlines)] #np.zeros((nlines, 3, lbw))
        
        for i,line in enumerate(reffreqs):
            
            ledge = int(utils.best_match_indx(line, self.x.compressed()) - lbw/2.)
            redge = int(utils.best_match_indx(line, self.x.compressed()) + lbw/2.)
            
            if ledge < 0 or nlines == 1:
                ledge = 0
            if redge > self.nx or nlines == 1:
                redge = self.nx
            
            splits[i][0].append(self.x.compressed()[ledge:redge])
            splits[i][1].append(self.y.compressed()[ledge:redge])
            splits[i][2].append(self.z.compressed()[ledge:redge])
            
        return splits
            
        
class Stack(object):
    """
    
    Construction::
    
    stack = Stack(line, transitions, specs)
    
    Parameters
    ----------
    line : :obj:`str`
          Line contained in the spectrum.
    transitions : array_like
          Transitions inside the Stack.
    specs : array of :obj:`Spectrum`
          Array with Spectrum objects to be stacked.
    """
    
    def __init__(self, line, transitions, specs, vmin=-100, vmax=100, dv=0):
        
        self.line = line
        self.transitions = transitions
        self.specs = specs
        self.vmin = vmin
        self.vmax = vmax
        if dv != 0:
            self.dv = dv
        else:
            self.compute_dv()
        self.x = np.ma.masked_invalid(np.arange(self.vmin, self.vmax, self.dv))
        self.y = np.ma.masked_invalid(np.zeros(len(self.x)))
        self.z = np.ma.masked_invalid(np.zeros(len(self.x)))
        self.model = np.ma.masked_invalid(np.ones(len(self.x)))
        
    def compute_dv(self):
        """
        Determines the maximum velocity separation
        between the channels inside Stack.specs.
        """
        
        for i,s in enumerate(self.specs):
            if i == 0:
                dv = s.dx
            else:
                dv = max(dv, s.dx)
        
        self.dv = dv
    
    def save(self, filename):
        """
        Saves the spectrum.
        
        Parameters
        ----------
        filename : :obj:`str`
                  Output filename.
        """
        
        np.savetxt(filename, np.c_[self.x.data, self.y.data, self.z.data])
    
    def stack_interpol(self, rms=True, ch0=0, chf=-1):
        """
        Stack by interpolating to a common grid.
        
        Parameters
        ----------
        rms : :obj:`bool`
             Compute a the rms for each substack?
        ch0 : :obj:`int`
             First channel used to compute the rms.
        chf : :obj:`int`
             Last channel used to compute the rms.
        """
        
        rms = np.zeros(len(self.specs))
        
        for i,spec in enumerate(self.specs):
        
            valid = np.ma.flatnotmasked_contiguous(spec.x)
            y_aux = np.zeros(len(self.x))
            z_aux = np.zeros(len(self.x))
            if not isinstance(valid, slice):
                for j,rng in enumerate(valid):
                    if len(spec.x[rng]) > 1:
                        interp_y = interpolate.interp1d(spec.x[rng], spec.y[rng],
                                                        kind='linear',
                                                        bounds_error=False,
                                                        fill_value=0.0)
                        y_aux += interp_y(self.x.data)
                        interp_z = interpolate.interp1d(spec.x[rng], spec.z[rng],
                                                        kind='linear',
                                                        bounds_error=False,
                                                        fill_value=0.0)
                        z_aux += interp_z(self.x.data)
                    elif not np.isnan(spec.x[rng]):
                        indx = utils.best_match_indx(spec.x[rng], self.x)
                        y_aux[indx] += spec.y[rng]
                        z_aux[indx] += spec.z[rng]
            else:
                interp_y = interpolate.interp1d(spec.x[valid], spec.y[valid],
                                                kind='linear',
                                                bounds_error=False,
                                                fill_value=0.0)
                y_aux += interp_y(self.x.data)
                interp_z = interpolate.interp1d(spec.x[valid], spec.z[valid],
                                                kind='linear',
                                                bounds_error=False,
                                                fill_value=0.0)
                z_aux += interp_z(self.x.data)
                
            # Check which channels have data
            ychan = [1 if ch != 0 else 0 for ch in y_aux]
            
            self.z += np.multiply(ychan, z_aux)
            self.y += y_aux*np.multiply(ychan, z_aux)
                        
            # Compute the rms up to this point
            rms[i] = (np.ma.masked_invalid(np.divide(self.y, self.z)))[ch0:chf].std()
            # Stacking, even a single line, reduces the rms. Is this due to the interpolation?
            
        # Divide by the total weight to preserve optical depth
        self.y = np.ma.masked_invalid(np.divide(self.y, self.z))
        
        # Apply a common mask
        self.mask = self.x.mask | self.y.mask | self.z.mask
        self.x.mask = self.mask
        self.y.mask = self.mask
        self.z.mask = self.mask
        
        return rms

def distribute_lines(lines, ngroups):
    """
    Groups a list of lines into ngroups.
    This is used to make stacks.
    
    Parameters
    ----------
    lines : array_like
           List of lines to group.
    ngroups : :ob:`int`
           Number of groups to make. If the number of lines is not divisible
           by the number of groups, then more lines are added to the first groups.
    """
    
    glens = [len(lines)/int(ngroups)]*ngroups

    i = 0
    while np.sum(glens) != len(lines):
        glens[i] += 1
        i += 1
    
    k = 0
    lgs = [[] for i in range(ngroups)]
    for line in lines:
        lgs[k].append(line)
        if len(lgs[k]) == glens[k]:
            k += 1
    
    return lgs

        
if __name__ == '__main__':
    import doctest
    doctest.testmod()
