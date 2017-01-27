#!/usr/bin/env python

import logging
import inspect

import numpy as np

from matplotlib import _cntr as cntr
from astropy.coordinates import Angle
from astropy import constants as c
from astropy import wcs

class Polygon:
    """
    Generic polygon class.

    Note: code based on:
    http://code.activestate.com/recipes/578381-a-point-in-polygon-program-sw-sloan-algorithm/

    Parameters
    ----------
    x : array
        A sequence of nodal x-coords.
    y : array
        A sequence of nodal y-coords.

    """
    def __init__(self, x, y):
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Creating Polygon")
        
        if len(x) != len(y):
            raise IndexError('x and y must be equally sized.')
        self.x = np.asfarray(x)
        self.y = np.asfarray(y)

        # Closes the polygon if were open
        x1, y1 = x[0], y[0]
        xn, yn = x[-1], y[-1]
        if x1 != xn or y1 != yn:
            self.x = np.concatenate((self.x, [x1]))
            self.y = np.concatenate((self.y, [y1]))

        # Anti-clockwise coordinates
        if _det(self.x, self.y) < 0:
            self.x = self.x[::-1]
            self.y = self.y[::-1]
            
    def get_vertices(self):
        """
        Returns the vertices of the polygon as a 2xNvert list.
        """
        
        return [[i,j] for i,j in zip(self.x, self.y)]

    def is_inside(self, xpoint, ypoint, smalld=1e-12):
        """
        Check if point is inside a general polygon.

        An improved version of the algorithm of Nordbeck and Rydstedt.

        REF: SLOAN, S.W. (1985): A point-in-polygon program. Adv. Eng.
             Software, Vol 7, No. 1, pp 45-47.

        Parameters
        ----------
        xpoint : array or float
            The x-coord of the point to be tested.
        ypoint : array or float
            The y-coords of the point to be tested.
        smalld : float
            Tolerance within which point is considered to be on a side.

        Returns
        -------
        mindst : array or float
            The distance from the point to the nearest point of the polygon:
                If mindst < 0 then point is outside the polygon.
                If mindst = 0 then point in on a side of the polygon.
                If mindst > 0 then point is inside the polygon.

        """
        xpoint = np.asfarray(xpoint)
        ypoint = np.asfarray(ypoint)

        # Scalar to array
        if xpoint.shape is tuple():
            xpoint = np.array([xpoint], dtype=float)
            ypoint = np.array([ypoint], dtype=float)
            scalar = True
        else:
            scalar = False
        # Check consistency
        if xpoint.shape != ypoint.shape:
            raise IndexError('x and y  must be equally sized.')

        # If snear = True: Dist to nearest side < nearest vertex
        # If snear = False: Dist to nearest vertex < nearest side
        snear = np.ma.masked_all(xpoint.shape, dtype=bool)

        # Initialize arrays
        mindst = np.ones_like(xpoint, dtype=float) * np.inf
        j = np.ma.masked_all(xpoint.shape, dtype=int)
        x = self.x
        y = self.y
        n = len(x) - 1  # Number of sides/vertices defining the polygon

        # Loop over each side defining polygon
        for i in range(n):
            d = np.ones_like(xpoint, dtype=float) * np.inf

            # Start of side has coords (x1, y1)
            # End of side has coords (x2, y2)
            # Point has coords (xpoint, ypoint)
            x1 = x[i]
            y1 = y[i]
            x21 = x[i+1] - x1
            y21 = y[i+1] - y1
            x1p = x1 - xpoint
            y1p = y1 - ypoint

            # Points on infinite line defined by
            #     x = x1 + t * (x1 - x2)
            #     y = y1 + t * (y1 - y2)
            # where
            #     t = 0    at (x1, y1)
            #     t = 1    at (x2, y2)
            # Find where normal passing through (xpoint, ypoint) intersects
            # infinite line
            t = -(x1p * x21 + y1p * y21) / (x21 ** 2 + y21 ** 2)
            tlt0 = t < 0
            tle1 = (0 <= t) & (t <= 1)

            # Normal intersects side
            d[tle1] = ((x1p[tle1] + t[tle1] * x21) ** 2 +
                       (y1p[tle1] + t[tle1] * y21) ** 2)

            # Normal does not intersects side
            # Point is closest to vertex (x1, y1)
            # Compute square of distance to this vertex
            d[tlt0] = x1p[tlt0] ** 2 + y1p[tlt0] ** 2

            # Store distances
            mask = d < mindst
            mindst[mask] = d[mask]
            j[mask] = i

            # Point is closer to (x1, y1) than any other vertex or side
            snear[mask & tlt0] = False

            # Point is closer to this side than to any other side or vertex
            snear[mask & tle1] = True

        if np.ma.count(snear) != snear.size:
            raise IndexError('Error computing distances')
        mindst **= 0.5

        # Point is closer to its nearest vertex than its nearest side, check if
        # nearest vertex is concave.
        # If the nearest vertex is concave then point is inside the polygon,
        # else the point is outside the polygon.
        jo = j.copy()
        jo[j==0] -= 1
        area = _det([x[j+1], x[j], x[jo-1]], [y[j+1], y[j], y[jo-1]])
        mindst[~snear] = np.copysign(mindst, area)[~snear]

        # Point is closer to its nearest side than to its nearest vertex, check
        # if point is to left or right of this side.
        # If point is to left of side it is inside polygon, else point is
        # outside polygon.
        area = _det([x[j], x[j+1], xpoint], [y[j], y[j+1], ypoint])
        mindst[snear] = np.copysign(mindst, area)[snear]

        # Point is on side of polygon
        mindst[np.fabs(mindst) < smalld] = 0

        # If input values were scalar then the output should be too
        if scalar:
            mindst = float(mindst)
        return mindst
    
    def make_mask(self, shape, **kwargs):
        """
        Creates a mask of a given shape using the Polygon as boundaries.
        All points inside the Polygon will have a value of 1.
        
        :param shape: Shape of the output mask.
        :type shape: tuple
        :returns: Mask of the Polygon.
        :rtype: array
        """
        
        mask = np.zeros(shape)
        
        xmax = int(round(max(self.x)))
        xmin = int(round(min(self.x)))
        ymax = int(round(max(self.y)))
        ymin = int(round(min(self.y)))
        
        for j in xrange(ymax - ymin):
            for i in xrange(xmax - xmin):
                if self.is_inside(i+xmin, j+ymin, **kwargs) >= 0:
                    self.logger.debug("Point ({0},{1}) ".format(i+xmin,j+ymin) +
                                      "is inside the Polygon")
                    mask[j+ymin,i+xmin] = 1
                    
        return mask

def _det(xvert, yvert):
    """
    Compute twice the area of the triangle defined by points using the
    determinant formula.

    Parameters
    ----------
    xvert : array
        A vector of nodal x-coords.
    yvert : array
        A vector of nodal y-coords.

    Returns
    -------
    area : float
        Twice the area of the triangle defined by the points:
            area is positive if points define polygon in anticlockwise order.
            area is negative if points define polygon in clockwise order.
            area is zero if at least two of the points are concident or if
            all points are collinear.

    """
    xvert = np.asfarray(xvert)
    yvert = np.asfarray(yvert)
    x_prev = np.concatenate(([xvert[-1]], xvert[:-1]))
    y_prev = np.concatenate(([yvert[-1]], yvert[:-1]))
    return np.sum(yvert * x_prev - xvert * y_prev, axis=0)

def beam_area_pix(head):
    """
    Computes the beam area in pixels.
    It uses an approximation accurate to
    within 5%.
    
    K. Rohlfs and T.L. Wilson, 'Tools of Radio Astronomy', third revised and enlarged edition, 1996, Springer, page 190-191.
    
    :param head: Image header.
    :type head: Fits header
    :returns: Number of pixels inside the beam.
    :rtype: float
    """
    
    return 1.133*float(head['BMAJ'])*float(head['BMIN'])/(abs(head['CDELT1'])*abs(head['CDELT2']))

def beam_area(head):
    """
    Computes the beam area in sterradians.
    It uses an approximation accurate to
    within 5%.
    
    K. Rohlfs and T.L. Wilson, 'Tools of Radio Astronomy', third revised and enlarged edition, 1996, Springer, page 190-191.
    
    :param head: Image header.
    :type head: Fits header
    :returns: Beam area in sr.
    :rtype: float
    """
    
    return np.pi/(4.*np.log(2.))*np.deg2rad(float(head['BMAJ']))*np.deg2rad(float(head['BMIN']))

def check_ascending(ra, dec, vel, verbose=False):
    """
    Check if the RA, DEC and VELO axes of a cube are in ascending order.
    It returns a step for every axes which will make it go in ascending order.
    
    :param ra: RA axis.
    :param dec: DEC axis.
    :param vel: Velocity axis.
    :returns: Step for RA, DEC and velocity.
    :rtype: int,int,int
    """
    
    if vel[0] > vel[1]: 
        vs = -1
        if verbose:
            print "Velocity axis is inverted."
    else:
        vs = 1
            
    if ra[0] > ra[1]:
        rs = -1
        if verbose:
            print "RA axis is inverted."
    else:
        rs = 1
        
    if dec[0] > dec[1]:
        ds = -1
        if verbose:
            print "DEC axis is inverted."
    else:
        ds = 1
        
    return rs, ds, vs

def compare_headers(head1, head2):
    """
    Compares the size and element width of 2 fits headers.
    """
    
    axes = np.array([False, False, False])
    
    for i in range(3):
        if head1['CDELT{0}'.format(i+1)] == head2['CDELT{0}'.format(i+1)] \
           and head1['NAXIS{0}'.format(i+1)] == head2['NAXIS{0}'.format(i+1)]:
               axes[i] = True
               
    if np.prod(axes) == 1:
        return True
    else:
        return False

def draw_beam(header, ax, **kwargs):
    """
    Draws an elliptical beam in a pywcsgrid2 axes object.
    """
    
    bmaj = header.get("BMAJ")
    bmin = header.get("BMIN")
    pa = header.get("BPA")
    pixx = header.get("CDELT1")
    pixy = header.get("CDELT2")
    ax.add_beam_size(bmaj/np.abs(pixx), bmin/np.abs(pixy), pa, loc=3, **kwargs)

def get_axis(header, axis):
    """
    Constructs a cube axis.
    
    :param header: Fits cube header.
    :type header: pyfits header
    :param axis: Axis to reconstruct.
    :type axis: int
    :returns: cube axis
    :rtype: numpy array
    """
    
    axis = str(axis)
    dx = header.get("CDELT" + axis)
    try:
        dx = float(dx)
        p0 = header.get("CRPIX" + axis)
        x0 = header.get("CRVAL" + axis)
        
    except TypeError:
        dx = 1
        p0 = 1
        x0 = 1

    n = header.get("NAXIS" + axis)
    
    p0 -= 1 # Fits files index start at 1, not for python.
    
    axis = np.arange(x0 - p0*dx, x0 - p0*dx + n*dx, dx)
    
    if len(axis) > n:
        axis = axis[:-1]
    
    return axis

def get_fits3axes(head):
    """
    """
    
    ra = get_axis(head, 1)
    de = get_axis(head, 2)
    ve = get_axis(head, 3)
    
    return ra , de, ve
    
def get_contours(x, y, z, levs, segment=0, verbose=False):
    """
    Creates an array with the contour vertices.
    """
    
    c = cntr.Cntr(x, y, z)
    
    segments = []
    for i,l in enumerate(levs):
        res = c.trace(l)
        if res:
            nseg = len(res) // 2
            segments.append(res[:nseg][segment])
            if verbose: print res[:nseg][segment]
        else:
            pass
                
    return np.asarray(segments)

def K2Jy(head):
    """
    Computes the conversion factor Jy/K.
    
    :param head: Image header.
    :type head: Fits header
    :returns: Factor to convert K to Jy.
    :rtype: float
    """
    
    omega = beam_area(head)
    k2jy = 2.*c.k_B.cgs.value/np.power(c.c.cgs.value/head['RESTFREQ'], 2.)*omega/1e-23
    
    return k2jy

def read_casa_polys(filename, image=None, wcs=None):
    """
    Reads casa region file and returns Polygon objects.
    
    Code adapted from FACTOR
    https://github.com/lofar-astron/factor/blob/reimage/factor/scripts/make_clean_mask.py
    https://github.com/lofar-astron/factor/commit/667b77e8690e9536a61afbe3ed673a3e16889bb1
    
    :param filename: Path to file containing the casa region.
    :type filename: string
    :param image: pyrap.images.image object, with properly defined coordsys.
    :type image: pyrap.images.image
    :returns: list of Polygons.
    :rtype: Polygon
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    polys = []
    for line in lines:
        if line.startswith('poly'):
            poly_str_temp = line.split('[[')[1]
            poly_str = poly_str_temp.split(']]')[0]
            poly_str_list = poly_str.split('], [')
            ra = []
            dec = []
            for pos in poly_str_list:
                RAstr, Decstr = pos.split(',')
                ra.append(Angle(RAstr, unit='hourangle').to('deg').value)
                dec.append(Angle(Decstr.replace('.', ':', 2), unit='deg').to('deg').value)

            # Convert to image-plane Polygon
            xvert = []
            yvert = []
            for RAvert, Decvert in zip(np.array(ra), np.array(dec)):
                
                if image:
                    try:
                        pixels = image.topixel([0, 1, Decvert*np.pi/180.0,
                                                RAvert*np.pi/180.0])
                    except:
                        pixels = image.topixel([1, 1, Decvert*np.pi/180.0,
                                                RAvert*np.pi/180.0])
                    xvert.append(pixels[2]) # x -> Dec
                    yvert.append(pixels[3]) # y -> RA
                    
                elif wcs:
                    pixels = wcs.all_world2pix([[RAvert, 
                                                 Decvert]], 0)[0]
                    xvert.append(pixels[0])
                    yvert.append(pixels[1])
                    
            polys.append(Polygon(xvert, yvert))

        elif line.startswith('ellipse'):
            ell_str_temp = line.split('[[')[1]
            if '], 0.0' not in ell_str_temp and '], 90.0' not in ell_str_temp:
                mod_log.error('Only position angles of 0.0 and 90.0 are supported for CASA '
                              'regions of type "ellipse"')
                sys.exit(1)
            if '], 0.0' in ell_str_temp:
                ell_str = ell_str_temp.split('], 0.0')[0]
                pa = 0
            else:
                ell_str = ell_str_temp.split('], 90.0')[0]
                pa = 90
            ell_str_list = ell_str.split('], [')

            # Ellipse center
            RAstr, Decstr = ell_str_list[0].split(',')
            ra_center = Angle(RAstr, unit='hourangle').to('deg').value
            dec_center = Angle(Decstr.replace('.', ':', 2), unit='deg').to('deg').value
            
            if image:
                pixels = image.topixel([0, 1, dec_center*np.pi/180.0,
                                        ra_center*np.pi/180.0])
                x_center = pixels[2] # x -> Dec
                y_center = pixels[3] # y -> RA
                
            elif wcs:
                pixels = wcs.all_world2pix([[ra_center, dec_center]], 0)[0]
                x_center = pixels[0] # x -> Dec
                y_center = pixels[1] # y -> RA

            # Ellipse semimajor and semiminor axes
            a_str, b_str = ell_str_list[1].split(',')
            a_deg = float(a_str.split('arcsec')[0])/3600.0
            b_deg = float(b_str.split('arcsec')[0])/3600.0
            
            if image:
                pixels1 = image.topixel([0, 1, (dec_center-a_deg/2.0)*np.pi/180.0,
                    ra_center*np.pi/180.0])
                a_pix1 = pixels1[2]
                pixels2 = image.topixel([0, 1, (dec_center+a_deg/2.0)*np.pi/180.0,
                    ra_center*np.pi/180.0])
                a_pix2 = pixels2[2]
            elif wcs:
                pixels1 = wcs.all_world2pix([[ra_center, dec_center-a_deg/2.0]], 0)[0]
                a_pix1 = pixels1[1]
                pixels2 = wcs.all_world2pix([[ra_center, dec_center+a_deg/2.0]], 0)[0]
                a_pix2 = pixels2[1]
                
            a_pix = abs(a_pix2 - a_pix1)
            ex = []
            ey = []
            for th in range(0, 360, 1):
                if pa == 0:
                    # semimajor axis is along x-axis
                    ex.append(a_pix * np.cos(th * np.pi / 180.0)
                        + x_center) # x -> Dec
                    ey.append(a_pix * b_deg / a_deg * np.sin(th * np.pi / 180.0) + y_center) # y -> RA
                elif pa == 90:
                    # semimajor axis is along y-axis
                    ex.append(a_pix * b_deg / a_deg * np.cos(th * np.pi / 180.0)
                        + x_center) # x -> Dec
                    ey.append(a_pix * np.sin(th * np.pi / 180.0) + y_center) # y -> RA
                    
            polys.append(Polygon(ex, ey))

        elif line.startswith('box'):
            poly_str_temp = line.split('[[')[1]
            poly_str = poly_str_temp.split(']]')[0]
            poly_str_list = poly_str.split('], [')
            ra = []
            dec = []
            for pos in poly_str_list:
                RAstr, Decstr = pos.split(',')
                ra.append(Angle(RAstr, unit='hourangle').to('deg').value)
                dec.append(Angle(Decstr.replace('.', ':', 2), unit='deg').to('deg').value)
            ra.insert(1, ra[0])
            dec.insert(1, dec[1])
            ra.append(ra[2])
            dec.append(dec[0])

            # Convert to image-plane Polygon
            xvert = []
            yvert = []
            for RAvert, Decvert in zip(np.array(ra), np.array(dec)):
                
                if image:
                    try:
                        pixels = image.topixel([0, 1, Decvert*np.pi/180.0,
                                                RAvert*np.pi/180.0])
                    except:
                        pixels = image.topixel([1, 1, Decvert*np.pi/180.0,
                                                RAvert*np.pi/180.0])
                    xvert.append(pixels[2]) # x -> Dec
                    yvert.append(pixels[3]) # y -> RA
                    
                elif wcs:
                    pixels = wcs.all_world2pix([[RAvert, Decvert]], 0)[0]
                    xvert.append(pixels[0])
                    yvert.append(pixels[1])
                    
            polys.append(Polygon(xvert, yvert))

        elif line.startswith('#'):
            pass

        else:
            mod_log.error('Only CASA regions of type "poly", "box", or "ellipse" are supported')
            sys.exit(1)

    return polys

def remove_nans(contours, indx=0):
    """
    Removes NaN elements from a contour list produced with get_contours().
    """
    
    mask = np.isnan(contours[indx][:,0])
    contours[indx] = contours[indx][~mask]
    
    return contours

def set_wcs(head):
    """
    Build a WCS object given the 
    spatial header parameters.
    """
    
    # Create a new WCS object. 
    # The number of axes must be set from the start.
    w = wcs.WCS(naxis=2)
    
    w.wcs.crpix = [head['CRPIX1'], head['CRPIX2']]
    w.wcs.cdelt = [head['CDELT1'], head['CDELT2']]
    w.wcs.crval = [head['CRVAL1'], head['CRVAL2']]
    w.wcs.ctype = [head['CTYPE1'], head['CTYPE2']]
    
    return w