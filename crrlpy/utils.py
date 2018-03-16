#!/usr/bin/env python

import re
import numpy as np

def alphanum_key(s):
    """ 
    Turn a string into a list of string and number chunks.
    
    :param s: String
    :returns: List with strings and integers.
    :rtype: list
    
    :Example:
    
    >>> alphanum_key('z23a')
    ['z', 23, 'a']
    
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def best_match_indx(value, array):
    """
    Searchs for the index of the closest entry to value inside an array.
    
    :param value: Value to find inside the array.
    :type value: float
    :param array: List to search for the given value.
    :type array: list or numpy.array
    :return: Best match index for the value inside array.
    :rtype: float
    
    :Example:
    
    >>> a = [1,2,3,4]
    >>> best_match_indx(3, a)
    2
    
    """
    
    array = np.array(array)
    subarr = abs(array - value)
    subarrmin = subarr.min()
        
    return np.where(subarr == subarrmin)[0][0]

def factors(n):
    """
    Decomposes a number into its factors.
    :param n: Number to decompose.
    :type n: int
    :return: List of values into which n can be decomposed.
    :rtype: list
    """

    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def flatten_list(list):
    """
    | Flattens a list of lists.
    | Based on:
    | http://stackoverflow.com/questions/457215/comprehension-for-flattening-a-sequence-of-sequences/5330178#5330178
    """
    
    #print 'List: {0}'.format(list)
    result = []
    extend = result.extend
    for l in list:
        extend(l)
        
    return result

def get_max_sep(array):
    """
    Get the maximum element separation in an array.
    
    Parameters
    ----------
    array :   array
              Array where the maximum separation is wanted.
    
    Returns
    -------
    max_sep : float
              The maximum separation between the elements in `array`.
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1,2,3,4,5,7])
    >>> get_max_sep(x)
    2
    """

    return max(abs(array[0:-1:2] - array[1::2]))

def get_min_sep(array):
    """
    Get the minimum element separation in an array.
    
    Parameters
    ----------
    array :   array
              Array where the minimum separation is wanted.
    
    Returns
    -------
    max_sep : float
              The minimum separation between the elements in `array`.
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1,2,3,4,5,7])
    >>> get_min_sep(x)
    1
    """

    return min(abs(array[0:-1:2] - array[1::2]))

def myround(x, base=5):
    """
    """
    
    return int(base * round(float(x)/base))

def natural_sort(list):
    """ 
    Sort the given list in the way that humans expect. \
    Sorting is done in place.
    
    :param list: List to sort.
    :type list: list
    
    :Example:
    
    >>> my_list = ['spec_3', 'spec_4', 'spec_1']
    >>> natural_sort(my_list)
    >>> my_list
    ['spec_1', 'spec_3', 'spec_4']
    """
    
    list.sort(key=alphanum_key)

def pow_notation(number, sig_fig=2):
    """
    Converts a number to scientific notation keeping sig_fig signitifcant figures.
    """
    
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a,b = ret_string.split("e")
    b = int(b) #removed leading "+" and strips leading zeros too.
    
    return r"10^{{{0}}}".format(b)

def sci_notation(number, sig_fig=2):
    """
    Converts a number to scientific notation keeping sig_fig signitifcant figures.
    """
    
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a,b = ret_string.split("e")
    b = int(b) #removed leading "+" and strips leading zeros too.
    
    return r"{0}\times10^{{{1}}}".format(a, b)

def str2bool(str):
    """
    Converts a string to a boolean value. The conversion is case insensitive.
    
    :param str: string to convert.
    :type str: string
    :returns: True if str is one of: "yes", "y", "true", "t" or "1".
    :rtype: bool
    """
    
    return str.lower() in ("yes", "y", "true", "t", "1")

rotated_labels = []
def text_slope_match_line(text, x, y, line, dindx=1):
    global rotated_labels

    # find the slope
    xdata, ydata = line.get_data()

    x1 = xdata[best_match_indx(x, xdata)]
    x2 = xdata[best_match_indx(x, xdata)+dindx]
    y1 = ydata[best_match_indx(y, ydata)]
    y2 = ydata[best_match_indx(y, ydata)+dindx]

    rotated_labels.append({"text":text, "line":line, "p1":np.array((x1, y1)), "p2":np.array((x2, y2))})

def tryint(str):
    """
    Returns an integer if `str` can be represented as one.
    
    :param str: String to check.
    :type str: string
    :returns: True is str can be cast to an int.
    :rtype: int
    """
    
    try:
        return int(str)
    except:
        return str

def update_text_slopes():
    global rotated_labels

    for label in rotated_labels:
        # slope_degrees is in data coordinates, the text() and annotate() functions need it in screen coordinates
        text, line = label["text"], label["line"]
        p1, p2 = label["p1"], label["p2"]

        # get the line's data transform
        ax = line.get_axes()

        sp1 = ax.transData.transform_point(p1)
        sp2 = ax.transData.transform_point(p2)

        rise = (sp2[1] - sp1[1])
        run = (sp2[0] - sp1[0])

        slope_degrees = np.rad2deg(np.arctan(rise/run))

        text.set_rotation(slope_degrees)
