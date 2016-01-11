#!/usr/bin/env python

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