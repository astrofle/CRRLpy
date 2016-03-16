#!/usr/bin/env python

import numpy as np

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
def text_slope_match_line(text, x, y, line):
    global rotated_labels

    # find the slope
    xdata, ydata = line.get_data()

    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]

    rotated_labels.append({"text":text, "line":line, "p1":np.array((x1, y1)), "p2":np.array((x2, y2))})

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