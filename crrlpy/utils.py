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
    """
    
    return str.lower() in ("yes", "true", "t", "1")