# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:03:16 2018

@author: weijl
"""

class World:
    def __init__(self,network,labels=[],values=[]):
        self.labels=labels
        self.values=values
        self.known=[]
        self.network=network
        self.probmass=self.get_probability_mass(self.labels,self.values)
        self.added_mass=0
        
    def add_mass(self,mass):
        '''
        Add external mass from a diminished world
        '''
        self.added_mass+=mass
    
    def update_world(self,network):
        self.network=network
        self.probmass=self.get_probability_mass(self.labels,self.values)
    
    def get_probability_mass(self,labels,values):
        '''
        Get the probability mass for this world
        NOTE: THIS CURRENTLY ONLY WORKS PROPERLY FOR D-SEPARATED SUBNETWORK WORLDS, IT WON'T THROW ERRORS BUT IT SHOULDN'T BE USED FOR FULL-NETWORK OPERATIONS
        @param labels: the labels for all the values that are known
        @type labels: list[string]
        @param values: the values for each node indicated in labels. These parameters' orders match.
        @type values: list[float]
        '''
        axis=labels[0]
        hp=self.network.nodes[axis+'_ant'].hparam
        pd=self.network.nodes[axis+'_ant'].hp_to_pd(self.network.nodes[axis+'_ant'].collapse_hp(labels))
        ind=[]
        for label,value in zip(labels,values):
            if label in hp[2]:
                i=hp[2].index(label)
                if i>0:
                    j=hp[1][i-1].index(value)
                else:
                    j=hp[0].index(value)
                ind.append(j)
        return pd[ind[0]][ind[1]][ind[2]][ind[3]]
        
    def get_similarity(self, other):
        '''
        Get the similarity measure between this world and a given other world.
        This function assumes that the worlds have shared history.
        @param other: the world to define a similarity to
        @type other: World
        '''
        sim=0
        for lab,val in zip(self.labels,self.values):
            for lab2,val2 in zip(other.labels,other.values):
                if lab in self.known and lab2 in other.known:
                    if lab==lab2 and val==val2:
                        sim+=1
        return sim
    
    def shared_history(self, other):
        '''
        determines whether this world and a given other world share the same history.
        @param other: the world to check a shared history for
        @type other: World
        '''
        for lab,val in zip(self.labels,self.values):
            if lab in self.known:
                # If the other.known.index fails, it means it has not filled in said value, and thus, the history will by definition not be shared.
                try:
                    i = other.known.index(lab)
                    if other.values[i]!=val:
                        return False
                except:
                    return False
        # Unless the try-except has failed, and no incongruencies have been found with the other world's past:
        return True
    
    def spread_probability_mass(self,others):
        '''
        Determine how this world's probability mass should be spread to given other worlds
        This function is hideous, but it works
        @param others: The surviving worlds that willr eceive some probability mass. 
                       These need not be pre-pruned to contain only worlds that receive mass from this world.
        @type others: list[World]
        '''
        similarities = []
        for other in others:
            if self.shared_history(other):
                sim = self.get_similarity(other)
                similarities.append(sim)
        
        total_probmass = self.probmass+self.added_mass
        probmasses=[]
        maxsim=max(similarities)
        
        # goodness gracious this is specific code
        # get the probability distirbution marginalized over the known variables
        axis=self.labels[0]
        hp=self.network.nodes[axis+'ant'].hparam
        hp=(hp[0],hp[1],[label for label in hp[2] if label not in self.known],hp[3])
        pd=self.network.nodes[axis+'ant'].hp_to_pd(self.network.nodes[axis+'ant'].collapse_hp([label for label in self.labels if label not in self.known]))
        
        for index,other in enumerate(others):
            if similarities[index]==maxsim:
                # if this other world is among the most similar, add its weighted share of the total probmass
                
                ind=[]
                otherprior=0
                
                #determine what indices we can find the other world's proportional probability
                for label,value in zip(other.labels,other.values):
                    if label not in self.known:
                        i=hp[2].index(label)
                        if i>0:
                            j=hp[1][i-1].index(value)
                        else:
                            j=hp[0].index(value)
                        ind.append(j)
                
                # get the prior probability for the given other world conditioned on known variables
                otherprior=1
                if len(ind)==1:
                    otherprior=pd[ind[0]]
                elif len(ind)==2:
                    otherprior=pd[ind[0]][ind[1]]
                elif len(ind)==3:
                    otherprior=pd[ind[0]][ind[1]][ind[2]]
                elif len(ind)==4:
                    otherprior=pd[ind[0]][ind[1]][ind[2]][ind[3]]

                weighted_pm=total_probmass*otherprior
                probmasses.append(weighted_pm)
            else:
                # otherwise, it gets none
                probmasses.append(0)
                
        return probmasses