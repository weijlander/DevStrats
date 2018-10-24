# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:04:34 2018

@author: Wouter Eijlander
"""

import numpy as np
import scipy.stats as stats
import math
from ArmModel import ArmModel
from EyeModel import EyeModel
from GenModel import GenModel
from t_distr import t_distr

class Infant():
    def __init__(self, drange=([-30,30],np.arange(-30,30.01,0.1))):
        '''
        drange: tuple ([min,max],[positions]) min and max position range, and the steps between those. 
                These indicate how each dimension of the space cube looks
        '''
        self.am = [0,0,0,0,0,0,0,0]
        self.em = [0,0,0,0,0]
        self.al = [[-20,130],[-20,70],[-70,60],[0,140]]
        self.el = [[-45,45],[5,50],[-45,45]]
        self.rArm = ArmModel(limits=self.al)
        self.eyes = EyeModel(limits=self.el)
        self.aGen = GenModel("2step.xdsl")
        self.eGen = GenModel("2step.xdsl")
        self.drange = drange
        self.cert = 0.95
        
    def fixate(self,target,hand=None):
        '''
        target: tuple ([centre_d],width) indicating centre positions for dimensions d and the overall width for the target
        hand:   tuple ([centre_d],width) indicating centre positions for dimensions d and the overall width for the hand
        '''
        targets=[target,hand] if hand else [target]
        space = t_distr(targets,self.drange)
        vis = self.eyes.process_inputs(self.eyes.dir,space)
        for it in range(10):
            if stats.entropy(vis[0])>(self.cert*stats.entropy(np.ndarray.tolist(np.ones(100)))):
                (mus,coac) = self.eGen.random(space)
            else:
                (mus,coac) = self.eGen.predict(vis,space)
            dir = self.eyes.move(mus,coac)
            vis = self.eyes.process_inputs(dir,space)
        return vis
    
    def reach(self,target):
        pass