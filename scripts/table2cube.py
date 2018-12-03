#!/usr/bin/env python

import sys

import numpy as np

def main(asciifile, outcube, datacol):
    """
    """
    
    data = np.loadtxt(asciifile)
    
    Te = np.unique(data[:,0])
    ne = np.unique(data[:,1])
    Tr = np.unique(data[:,2])
    
    cube = np.empty((len(Tr), len(ne), len(Te)))
    
    for i in range(len(Tr)):
        for j in range(len(ne)):
            for k in range(len(Te)):
                                
                mask = (data[:,2] == Tr[i]) & (data[:,1] == ne[j]) & (data[:,0] == Te[k])
                
                if len(np.where(mask == True)[0]) > 0:
                    #print 'added'
                    cube[i,j,k] = data[mask,datacol]
                else:
                    cube[i,j,k] = np.nan
    
    #np.save(outcube, )

if __name__ == '__main__':
    
    logfile = sys.argv[1]
    outcube = sys.argv[2]
    datacol = int(sys.argv[3])
    
    #main(logfile, outcube, datacol)
    
    data = np.loadtxt(logfile)
    
    te = np.unique(data[:,0])
    ne = np.unique(data[:,1])
    tr = np.unique(data[:,2])
    
    cube = np.empty((len(tr), len(ne), len(te), 4))
    
    for i in range(len(tr)):
        for j in range(len(ne)):
            for k in range(len(te)):
                                
                mask = (data[:,2] == tr[i]) & (data[:,1] == ne[j]) & (data[:,0] == te[k])
                
                if len(np.where(mask == True)[0]) > 0:
                    #print 'added'
                    cube[i,j,k,0] = data[mask,2][0]
                    cube[i,j,k,1] = data[mask,1][0]
                    cube[i,j,k,2] = data[mask,0][0]
                    cube[i,j,k,3] = data[mask,datacol][0]
                else:
                    cube[i,j,k] = np.nan
                    
    np.save(outcube, cube)
    