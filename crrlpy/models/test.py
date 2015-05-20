#!/usr/bin/env python

import crrlpy.models.rrlmod as rm
import pylab as plt

salg = rm.load_betabn('7d1', '1d0', other='')
data = rm.make_betabn('7d1', '1d0', 'alpha', nmax=1000, other='')

plt.plot(salg[100:1000,0], salg[100:1000,1], 'b-')
plt.plot(data[0,100:1000], data[1,100:1000], 'ro')

plt.show()