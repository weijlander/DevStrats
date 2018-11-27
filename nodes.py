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
            self.hparam=self.con_hparam(self.parents)
        else:
            self.hparam=self.con_hparam()
        
    def con_hparam(self,parents=None):
        '''
        construct the hyperparameter of the node for a given parent, or construct a hyperprior if no parent is given
        @param parent: the parent node with which to construct a hyperparameter
        @type parent: Node
        '''
        if parents:
            shape = [len(self.values)]
            for p in parents:
                shape.append(len(p.values))
            shape=tuple(shape)
            hparam = (self.values, [p.values for p in parents], [self.label]+[p.label for p in parents], np.ndarray.tolist(np.zeros(shape)))
        else:
            hparam = (self.values, [], [], np.ndarray.tolist(np.zeros(len(self.values))))
        return hparam
    
    def update_hp(self,labels,values):
        '''
        update the node's hyperparameters given a new observation
        @param labels: the labels for the nodes in the observation
        @type labels: list[String]
        @param values: the values for the nodes in the observation
        @type values: list[float]
        '''
        own_v = values[labels.index(self.label)]
        own_b = self.get_bin(own_v)
        ind_s = self.values.index(own_b)
        if self.parents:
            indices=[ind_s]
            for p in self.parents:
                par_v = values[labels.index(p.label)]
                par_b = p.get_bin(par_v)
                ind_p = p.values.index(par_b)
                indices.append(ind_p)
            self.add_to_index(indices)
        else:
            self.hparam[3][ind_s]+=1
    
    def get_p(self,value):
        '''
        get the probability from the bin that the given value resides in
        @param value: the value of interest
        @type value: float
        '''
        b=self.get_bin(value)
        return b[1]
        
    def get_bin(self,value):
        '''
        # returns the bin that the given value resides in
        @param value: the value of interest
        @type value: float
        '''
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
 
    def collapse_hp(self,queries):
        '''
        Collapse the hyperparameter to form a joint occurrence distribution that contains only the queried variables.
        NOTE: THIS IS NOT A PROBABILITY DISTRIBUTION. TO MAKE A JPD FROM THIS, CALL THE GENERIC CLASS METHOD hp_to_pd
        @param queries: the queried variable labels. these must be labels that are used by the node since it doesn't check for that
        @type queries: list[string]
        '''
        labels=self.hparam[2]
        axes=tuple([self.hparam[2].index(i) for i in labels if i not in queries])
        return np.sum(self.hparam[3],axis=axes)
    
    def hp_to_pd(self,pdistr):
        return np.ndarray.tolist(np.divide(pdistr,sum(pdistr)+0.01))

###############################################################################
####       SUBCLASSES
###############################################################################

class Axis(Node):
    def update_hp(self,labels,values,nodes):
        '''
        update the node's hyperparameters given a new observation
        @param labels: the labels for the nodes in the observation
        @type labels: list[String]
        @param values: the values for the nodes in the observation
        @type values: list[float]
        '''
        own_v = values[labels.index(self.label)]
        own_b = self.get_bin(own_v)
        ind_s = self.values.index(own_b)
        self.hparam[3][ind_s]+=1
        
    def update_pd(self):
        '''
        update the node's pdistr to reflect the new values in its hyperparameters
        '''
        counts_per_value = self.hparam[-1]
        new_pdistr = np.ndarray.tolist(np.divide(counts_per_value,sum(counts_per_value)))
        self.pdistr = new_pdistr
    
class Coeff(Node):
    
    def update_hp(self,labels,values,nodes):
        '''
        update the node's hyperparameters given a new observation
        @param labels: the labels for the nodes in the observation
        @type labels: list[String]
        @param values: the values for the nodes in the observation
        @type values: list[float]
        '''
        own_v = values[labels.index(self.label)]
        own_b = self.get_bin(own_v)
        ind_s = self.values.index(own_b)
        ind=[ind_s]
        for p in self.parents:
            par_v = values[labels.index(p.label)]
            par_b = p.get_bin(par_v)
            ind_p = p.values.index(par_b)
            ind.append(ind_p)
        self.hparam[3][ind[0]][ind[1]][ind[2]][ind[3]][ind[4]]+=1
        
    def update_pd(self):
        '''
        update the node's pdistr to reflect the new values in its hyperparameters
        '''
        #counts_per_value = [sum(self.hparam[3][-1][-1][-1][i]) for i in range(len(self.values))]
        counts_per_value = self.collapse_hp([self.label])
        new_pdistr = self.hp_to_pd(counts_per_value)
        self.pdistr = new_pdistr
        
#    def get_cpt(self,parent_label):
#        '''
#        Get the conditional probability table for this node given a parent label
#        '''
#        parent_index = self.hparam[2].index(parent_label)
#        for i in [1,2,3]:
#            cpt = np.ndarray.tolist(np.divide(hp[-1],np.sum(hp[-1])))
#            return (hp[0],hp[1],hp[2],cpt)

class Musc(Node):
    def extend_hp(self,nodes):
        '''
        Extend the node's hyperparameter to include values of the parents' other children (causally these are actually the childrens' other parents)
        @param nodes: all the nodes in the network to search through for shared parentage
        @type nodes: list[Node]
        '''
        for parent in self.parents:
            for node in nodes:
                if parent in node.parents and node.label != self.label:
                    self.hparam[1].append(node.values)
                    self.hparam[2].append(node.label)
                    # for some reason the following 3 lines can't be contracted into 1
                    shape = list(np.shape(self.hparam[3]))
                    shape.append(10)
                    shape=tuple(shape)
                    self.hparam = self.hparam [:3] + tuple([np.ndarray.tolist(np.zeros(shape))])
                
    
    def update_hp(self,labels,values,nodes):
        '''
        update the node's hyperparameters given a new observation
        @param labels: the labels for the nodes in the observation
        @type labels: list[String]
        @param values: the values for the nodes in the observation
        @type values: list[float]
        '''
        own_v = values[labels.index(self.label)]
        own_b = self.get_bin(own_v)
        ind_s = self.values.index(own_b)
        ind=[ind_s]
        for p in self.hparam[2]:
            par_v = values[labels.index(p)]
            par_b = nodes[p].get_bin(par_v)
            ind_p = nodes[p].values.index(par_b)
            ind.append(ind_p)
        self.hparam[3][ind[0]][ind[1]][ind[2]][ind[3]]+=1
        
    def update_pd(self):
        '''
        update the node's pdistr to reflect the new values in its hyperparameters
        '''
        counts_per_value = self.collapse_hp([self.label])
        new_pdistr = self.hp_to_pd(counts_per_value)
        self.pdistr = new_pdistr
                    
#    def get_cpt(self,parent_label):
#        '''
#        Get the conditional probability table for this node given a parent label
#        '''
#        parent_index = self.hparam[2].index(parent_label)
#        for i in [1,2,3]:
#            cpt = np.ndarray.tolist(np.divide(hp[-1],np.sum(hp[-1])))
#            return (hp[0],hp[1],hp[2],cpt)