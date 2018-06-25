# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:04:34 2018

@author: Wouter Eijlander
"""

import numpy as np
import math
from ArmModel import ArmModel
from EyeModel import EyeModel
from t_distr import t_distr

class Infant():
    def __init__(self, drange=([-30,30],np.arange(-30,30.01,0.1))):
        self.am = [0,0,0,0,0,0,0,0]
        self.em = [0,0,0,0,0]
        self.al = [[-10,130],[-10,100],[-90,90]]
        self.el = [[-45,45],[5,50],[-45,45]]
        self.rArm = ArmModel(limits=self.al)
        self.eyes = EyeModel(limits=self.el)
        self.aGen = "TODO: GENERATIVE MDOEL FOR ARM"
        self.eGen = "TODO: GENERATIVE MODEL FOR EYES"
        self.drange = drange
        
    def fixate(self,target):
        space = t_distr(target,self.drange)
        vis = self.eyes.process_inputs(self.eyes.dir,space)
        for it in range(10):
            (mus,coac) = self.eGen.predict(vis,space)
            dir = self.eyes.move(mus,coac)
            vis = self.eyes.process_inputs(dir,space)
        return vis