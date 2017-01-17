#!/usr/bin/env python

import os
import pickle
import numpy as np
from crrlpy.models import rrlmod

if __name__ == '__main__':
    
    n_min = 5
    n_max = 1000
    out = 'RRL_CIalpha_itau_dne'
    
    #Te = np.array(['1d1', '1.5d1', '2d1', '2.5d1', '3d1', '3.5d1', '4d1', '4.5d1', 
                   #'5d1', '5.5d1', '6d1', '6.5d1', '7d1', '7.5d1', '8d1', '8.5d1', 
                   #'9d1', '9.5d1', '1d2', '1.05d2', '1.1d2', '1.15d2', '1.2d2', 
                   #'1.25d2', '1.3d2', '1.35d2', '1.4d2', '1.45d2', '1.5d2', '1.55d2', 
                   #'1.6d2', '1.65d2', '1.7d2', '1.75d2', '1.8d2', '1.85d2', '1.9d2', 
                   #'1.95d2', '2d2', '2.05d2', '2.1d2', '2.15d2', '2.2d2', '2.25d2',  
                   #'2.3d2', '2.35d2', '2.4d2', '2.45d2', '2.5d2', '2.55d2', '2.6d2', 
                   #'2.65d2', '2.7d2', '2.75d2', '2.8d2', '2.85d2', '2.9d2', '2.95d2', 
                   #'3d2', '3.05d2', '3.1d2', '3.15d2', '3.2d2', '3.25d2', '3.3d2',
                   #'3.35d2', '3.4d2', '3.45d2', '3.5d2', '3.55d2', '3.6d2', '3.65d2',
                   #'3.7d2', '3.75d2', '3.8d2', '3.85d2', '3.9d2', '3.95d2', '4d2'])
    Te = np.array(['1d1', '1.5d1', '2d1', '2.5d1', '3d1', '3.5d1', '4d1', '4.5d1', 
                   '5d1', '5.5d1', '6d1', '6.5d1', '7d1', '7.5d1', '8d1', '8.5d1', 
                   '9d1', '9.5d1', '1d2', '1.05d2', '1.1d2', '1.15d2', '1.2d2', 
                   '1.25d2', '1.3d2', '1.35d2', '1.4d2', '1.45d2', '1.5d2'])
    #ne = np.concatenate((np.arange(0.01, 0.11, 0.005), np.arange(0.011, 0.015, 0.001), np.arange(0.016, 0.02, 0.001)))
    ne = np.arange(0.01, 0.115, 0.005)
    Tr = np.array([800, 1200, 1400, 1600, 2000])
    
    models = rrlmod.models_dict(Te, ne, Tr)
    
    # Check if a models file already exists to avoid loading all the models again
    if not os.path.isfile('{0}.npy'.format(out)): 
    
        itau_mod = rrlmod.load_itau_dict(models, 'RRL_CIalpha', n_min=n_min, n_max=n_max, 
                                         verbose=False, value='itau')
        np.save('{0}.npy'.format(out), itau_mod)
        pickle.dump(models, open('{0}.p'.format(out), "wb"))
    else:
        itau_mod = np.load('{0}.npy'.format(out))
