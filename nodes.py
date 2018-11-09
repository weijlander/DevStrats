# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:14:51 2018

@author: Woutere Eijlander
"""

import numpy as np
     
class Node():
    # Archetypical node used in the prediction networks
    def __init__(self,label,parents,value=0.0,card=10):
        self.label=label
        self.value=value
        self.parents=parents
        self.values=[]
        self.pdistr=[]
        for step in range(card):
            self.values.append((1/(card-1))*step)
            self.pdistr.append(1/card)
        if self.parents:
            self.hparams=[self.con_hparam(p) for p in self.parents]
        else:
            self.hparams=[self.con_hparam()]
        
    def con_hparam(self,parent=None):
        if parent:
            return (self.values, parent.values, parent.label, np.ndarray.tolist(np.zeros((len(self.values),len(parent.values)))))
        else:
            return (self.values, [], '', np.ndarray.tolist(np.zeros(len(self.values))))
    
    def update_hp(self,labels,values):
        own_v = values[labels.index(self.label)]
        own_b = self.get_bin(own_v)
        ind_s = self.values.index(own_b)
        for hp in self.hparams:
            if self.parents:
                for p in self.parents:
                    par_v = values[labels.index(p.label)]
                    par_b = p.get_bin(par_v)
                    ind_p = p.values.index(par_b)
                    if hp[2] == p.label:
                        hp[3][ind_s][ind_p]+=1
            else:
                hp[3][ind_s]+=1
                    
    def get_p(self,value):
    # get the probability from the bin that the given value resides in
        b=self.get_bin(value)
        return b[1]
    
    def get_bin(self,value):
        # returns the bin that the given value resides in
        try:
            for val in range(len(self.pdistr)):
                if value>=self.values[val] and value < self.values[val+1]:
                    # determine the median between two bin values and check where the given value sits
                    med=(self.values[val]+self.values[val+1])/2
                    if value<=med:
                        return self.values[val]
                    else:
                        return self.values[val+1]
                elif value<self.values[val]:
                    return self.values[0]
        except:
            return self.values[-1]
        
##############################
# LEGACY CODE BELOW
##############################

#class Positional():
#    # Root nodes encodin the position of the target
#    def __init__(self,label,value=0.0):
#        self.label=label
#        self.value=value
#
#class Axis():
#    # The intermediate variables encoding activation of an individual axis
#    def __init__(self,label,parents, value=0.0, card=10):
#        self.label=label
#        self.parents=parents
#        self.value=value
#        self.values=[]
#        self.pdistr=[]
#        for step in range(card):
#            self.values.append((1/(card-1))*step)
#            self.pdistr.append(1/card)
#            
#    def get_p(self,value):
#    # get the probability from the bin that the given value resides in
#        b=self.get_bin(value)
#        return b[1]
#    
#    def get_bin(self,value):
#        # returns the bin that the given value resides in
#        try:
#            for val in range(len(self.pdistr)):
#                if value>=self.values[val] and value < self.values[val+1]:
#                    # determine the median between two bin values and check where the given value sits
#                    med=(self.values[val]+self.values[val+1])/2
#                    if value<=med:
#                        return self.pdistr[val]
#                    else:
#                        return self.pdistr[val+1]
#                elif value<self.values[val]:
#                    return self.pdistr[0]
#        except:
#            return self.pdistr[-1]
#        
#class Leaf():
#    # Prediction nodes encoding muscle activation or coactivation coefficient
#    def __init__(self,label,parents,value=0.0,card=10):
#        self.label=label
#        self.value=value
#        self.parents=parents
#        self.values=[]
#        self.pdistr=[]
#        for step in range(card):
#            self.values.append((1/(card-1))*step)
#            self.pdistr.append(1/card)
#            
#    def get_p(self,value):
#    # get the probability from the bin that the given value resides in
#        b=self.get_bin(value)
#        return b[1]
#    
#    def get_bin(self,value):
#        # returns the bin that the given value resides in
#        try:
#            for val in range(len(self.pdistr)):
#                if value>=self.values[val] and value < self.values[val+1]:
#                    # determine the median between two bin values and check where the given value sits
#                    med=(self.values[val]+self.values[val+1])/2
#                    if value<=med:
#                        return self.pdistr[val]
#                    else:
#                        return self.pdistr[val+1]
#                elif value<self.values[val]:
#                    return self.pdistr[0]
#        except:
#            return self.pdistr[-1]
#   