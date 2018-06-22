# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:14:49 2018

@author: Wouter Eijlander
"""

import numpy as np

def t_distr(target,drange):
    centre,width = target
    stepsize = len(drange[1])/(drange[0][1]-drange[0][0])
    # Determine index positions for the target centre, and its left/right borders
    c = int(stepsize*(drange[0][1]+centre))
    l = int(c-stepsize*width/2)
    r = int(c+stepsize*width/2)
    
    # determine the distribution's object height
    p = 1/(width*stepsize)
    
    t = np.zeros((len(drange[1]),1))
    for i in np.arange(l,r):
        t[i] = p
        
    return np.ndarray.tolist(t)