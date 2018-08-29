# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:14:49 2018

@author: Wouter Eijlander
"""

import numpy as np

def t_distr(target,drange,type="uniform"):
    centre,width = target
    stepsize = len(drange[1])/(drange[0][1]-drange[0][0])
    tar = []
	if type=="uniform":
		for d in range(len(centre)):
			# Determine index positions for the target centre, and its left/right borders
			c = int(stepsize*(drange[0][1]+centre[d]))
			l = int(c-stepsize*width/2)
			r = int(c+stepsize*width/2)
		   
			# determine the distribution's object height
			p = 1/(width*stepsize)
			t = np.zeros((len(drange[1]),1))
			t = np.ndarray.tolist(t)
			for i in np.arange(l,r):
				try:
					t[i-1] = p
				except:
					pass
			tar.append(t)
    elif type=="gaussian":
		
	elif type=="sampled":
	
	else:
		tar = np.ndarray.tolist(np.zeros(len(drange[1]))+1/len(drange[1]))
	
    return tar