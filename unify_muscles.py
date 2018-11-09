# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:20:33 2018

@author: Wouter Eijlander
"""
import random as r
import math

def unify_muscles(ag,ant,coac):
    '''
    ag, ant:    float, indicating muscle activities for opposing muscles
    coac:       float, coactivation coefficient
    '''
    R = (r.random()-0.5)/20
    rc = R/math.sqrt(coac+0.001)
    a = ag-ant*coac
    return a+a*rc