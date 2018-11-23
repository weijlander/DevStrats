# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:14:49 2018

@author: Wouter Eijlander
"""

import numpy as np
from scipy.stats import norm
from heapq import *

def t_distr(targets,drange,type="sampled_gauss"):
    '''  
    target: tuple ([centre_d],width) indicating centre positions for dimensions d and the overall width for the target
    drange: tuple ([min,max],[positions]) min and max position range, and the steps between those. 
            These indicate how each dimension of the space cube looks
    type:   string, indicating type, "uniform", "gaussian", "sampled_gauss" or "sampled_uni".
    '''
    
    stepsize =int( len(drange[1])/(drange[0][1]-drange[0][0]))
    tar = []
    for t in targets:
        centre,width = t
        if type=="uniform":
            for d in range(len(centre)):
                c,l,r = det_target(drange,stepsize,centre,width,d)
                # determine the distribution's object height
                p = 1/(width*stepsize)
                t = np.zeros((len(drange[1]),1))
                t = np.ndarray.flatten(t)
                t = np.ndarray.tolist(t)
                for i in np.arange(l,r):
                    try:
                        t[i-1] = p
                    except:
                        pass
                tar.append(t)
                
        elif type=="gaussian":
            for d in range(len(centre)):
                c,l,r = det_target(drange,stepsize,centre,width,d)
                # construct a gaussian G to fit between l and r centered around c
                g = norm.pdf(drange[1],centre[d],width/4)
                # put g into tar
                tar.append(g)
                
        elif type=="sampled_gauss":
            tdist = t_distr([t],drange,type="gaussian")
            for tdim in tdist:
                tdim = [float(i)/sum(tdim) for i in tdim]
                dim = list(drange[1])
                cur = np.ndarray.tolist(np.zeros(len(drange[1])))
                sample = np.random.choice(dim,size=len(drange[1]),p=tdim)
                for each in sample:
                    cur[dim.index(each)]+=1/len(sample)
                tar.append(cur)
                
        elif type=="sampled_uni":
            tdist = t_distr([t],drange,type="uniform")
            for tdim in tdist:
                tdim = [float(i)/sum(tdim) for i in tdim]
                dim = list(drange[1])
                cur = np.ndarray.tolist(np.zeros(len(drange[1])))
                sample = np.random.choice(dim,size=len(drange[1]),p=tdim)
                for each in sample:
                    cur[dim.index(each)]+=1/len(sample)
                tar.append(cur)
        else:
            tar = np.ndarray.tolist(np.zeros(len(drange[1]))+1/len(drange[1]))
        
    return tar

def heapsort(iterable):
    h=[]
    for value in iterable:
        heappush(h,value)
    return [heappop(h) for i in range(len(h))]

def det_target(drange,stepsize,centre,width,it):
    # Determine index positions for the target centre, and its left/right borders
    c = int(stepsize*(drange[0][1]+centre[it]))
    l = int(c-stepsize*width/2)
    r = int(c+stepsize*width/2)
    return c,l,r