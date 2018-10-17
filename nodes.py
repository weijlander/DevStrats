# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:14:51 2018

@author: Woutere Eijlander
"""

class Positional():
    # Root nodes encodin the position of the target
    def __init__(self,label,value=0.0):
        self.label=label
        self.value=value

class Axis():
    # The intermediate variables encoding activation of an individual axis
    def __init__(self,label,parents, value=0.0):
        self.label=label
        self.parents=parents
        self.value=value
        
class Leaf():
    # Prediction nodes encoding muscle activation or coactivation coefficient
    def __init__(self,label,parents,value=0.0):
        self.label=label
        self.value=value
        self.parents=parents