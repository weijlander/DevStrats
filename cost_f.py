# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:36:26 2018

@author: Wouter Eijlander
"""

def cost_f(muscles, coac):
    '''
    muscles:    list of muscle activations
    coac:       float, coactivation coefficient
    '''
    cost = 0
    temp = [x for x in range(len(muscles)) if x%2==0]
    for i in temp:
        try:
            ag = max(muscles[i],muscles[i+1])
            ant = min(muscles[i],muscles[i+1])
            cost+=(ag+coac*ant)
        except:
            break
    return cost