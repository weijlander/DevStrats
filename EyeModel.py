# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:03:38 2018

@author: Wouter Eijlander
"""

from move_eyes import move_eyes
import numpy as np
from scipy.stats import norm
import math

class EyeModel():
    def __init__(self,default=[0,5,0],limits=[[-45,45],[5,50],[-45,45]],vnoise=1,fovd=50,drange=np.arange(-30,30.01,0.1)):
        self.default = default
        self.limits = limits
        self.vnoise = vnoise
        self.fovd = fovd
        self.drange = drange
        self.dir = default
        self.drange = drange
    
    def move(self,muscles,coac):
        # moves the model's gaze direction, and returns it for convenience
        self.dir = move_eyes(self.default,muscles,self.limits,coac)
        return self.dir
    
    def makeKernel(self,d):
        # returns the x,y, and z-axis pdfs of the current fixation points
        len = np.linalg.norm(d)
        wid = len*(math.asin(math.radians(self.fovd))/math.acos(math.radians(self.fovd)))
        k = []
        for i in range(3):
            pd = norm(d[i],wid/3)
            j = [pd.pdf(x) for x in self.drange]
            k.append(j)
        return k
    
    def process_inputs(self,d,space):
        # TODO: ADD VISUAL NOISE
        # get x,y, and z-axis values for the target, hand, and the distributions for the visual kernel
        ker = self.makeKernel(d)
        try:
            st = space[0:3][:]
            sh = space[3:6][:]
            
            # get the overlap between the objects and visual fixation kernel
            t = []
            h = []
            for i in  range(3):
                t = np.append(t, [st[i][j]*ker[i][j] for j in range(len(ker))])
                h = np.append(t, [sh[i][j]*ker[i][j] for j in range(len(ker))])
            vision = (t,h)
        except:
            st = space[0:3][:]
            
            # get the overlap between the objects and visual fixation kernel
            t = []
            for i in  range(3):
                t = np.append(t, [st[i][j]*ker[i][j] for j in range(len(ker))])
            vision = t
        return vision