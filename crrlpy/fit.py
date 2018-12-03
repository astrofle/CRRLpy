#!/usr/bin/env python

def fit_itau(obs, mod):
    """
    """
    
    nindx = np.where(mod.prop['n'] - obs.prop['n'] == 0)
        
    mod_itau = mod.prop['itau'][nindx]
    obs_itau = obs.prop['itau']
    
    em = np.linalg.lstsq(obs_itau, mod_itau)[0]
    
    return em

if __name__ == '__main__':
    pass