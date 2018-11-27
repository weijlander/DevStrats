# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:04:34 2018

@author: Wouter Eijlander
"""

import numpy as np
import scipy.stats as stats
import itertools
from ArmModel import ArmModel
from t_distr import t_distr
from unify_muscles import unify_muscles
from worlds import World
from move_arm import *
from ArmNet import *
from tqdm import tqdm

class Infant():
    def __init__(self, name='Benjamin', drange=([-30,30],np.arange(-30,30.01,0.1)),card=3):
        '''
        drange: tuple ([min,max],[positions]) min and max position range, and the steps between those. 
                These indicate how each dimension of the space cube looks
        '''
        self.al = [[-20,130],[-20,70],[-70,60],[0,140]]
        self.rArm = ArmModel(limits=self.al)
        self.anet = armNet(name=name,card=card)
        self.drange = drange
        self.kbase = [[self.anet.nodes[n].value for n in self.anet.nodes]]
    
    def reach(self,target):
        '''
        @param target: the target position in 3d and its width
        @type target: tuple(list[x,y,z],float)
        '''
        # Determine the target position that the agent sees (this can differ from the actual target centre)
        target_distr = t_distr(target,self.drange)
        sampled_points = np.ndarray.tolist(np.amax(target_distr,axis=1))
        (x,y,z) = target_distr[0].index(sampled_points[0]),target_distr[1].index(sampled_points[1]),target_distr[2].index(sampled_points[2])
        target_pos = self.drange[1][x],self.drange[1][y],self.drange[1][z]
        
        # Determine the values for all the nodes in the agent's arm network by inference
        nodes=self.infer_nodes(target_pos)
        
        # Update beliefs in the network
        self.update_hparams([n for n in self.anet.nodes],nodes)
        for n in self.anet.nodes:
            self.anet.nodes[n].update_pd()
        pass
    
    def motor_babbling(self,nb=1000,type='gaussian'):
        '''
        build a knowledge base for the probability distributions over network's nodes
        @param nb: the number of random movements
        @type nb: int
        @param type: the type of distributions from which the leaf nodes in the network will be sampled: 'gaussian' or 'uniform', defaults to gaussian
        @type type: string
        '''
        for cycle in range(nb):
            values=[]
            # sample random muscle activations based on the babbling type
            for muscle in self.anet.muscles:
                # sample a random muscle activation and clip
                ranm=np.random.normal(loc=0.5,scale=0.25)
                ranm=min(max(ranm,0),1)
                values.append(ranm)
            # sample a random cc and clip
            rancc = np.random.normal(loc=0.5,scale=0.25)
            rancc=min(max(rancc,0),1)
            values.append(rancc)
            
            # determine the axis values given the random muscles and cc
            shx = unify_muscles(values[0],values[1],values[-1])
            shy = unify_muscles(values[2],values[3],values[-1])
            shz = unify_muscles(values[4],values[5],values[-1])
            elx = unify_muscles(values[6],values[7],values[-1])
            
            # determine the end-effector position given the random muscles and cc
            #X,Y,Z=0,0,0#move_arm(self.rArm.arm,values[:-1],self.rArm.limits,rancc)
            
            # add the new random findings to the knowledge base
            #nodes=[X,Y,Z]
            nodes=[shx,shy,shz,elx]
            nodes.extend(values)
            for i,n in enumerate (self.anet.nodes):
                b=self.anet.nodes[n].get_bin(nodes[i])
                nodes[i]=b
            self.kbase.append(nodes)
            self.update_hparams([n for n in self.anet.nodes],nodes)
        for n in self.anet.nodes:
            self.anet.nodes[n].update_pd()
    
    def infer_nodes(self,target):
        angles=inverse_approx(target,self.rArm.arm,angles=self.rArm.angles)
        shx,shy,shz,elx=self.rArm.angle_to_activation(angles)
        nodes=self.imaging([shx,shy,shz,elx])
        return nodes
    
    def imaging(self,axes):
        '''
        @param axes: the activations for the axis nodes that have been pre-calculated
        @type target: list[float]
        '''
        worlds=self.get_worlds()
          
        return nodes
    
    def get_worlds(self):
        labels=[l for l in self.anet.nodes]
        nworlds=pow(len(self.anet.nodes['shx'].values),len(labels))
        vals=[self.anet.nodes['shx'].values for node in labels]
        values=list(itertools.product(*vals))
        
        worlds=[]
        for wn in tqdm(range(nworlds)):
            #determine world i's value set, and make the world
            v=list(values[0])
            worlds.append(World(self.anet,labels,v))
            values=values[1:]
        return worlds
    
    def update_hparams(self,labels,values):
        '''
        @param labels: the labels of all the nodes
        @type labels: list[string]
        @param values: the values for all the nodes
        @type values: float
        '''
        for n in self.anet.nodes:
            self.anet.nodes[n].update_hp(labels,values,self.anet.nodes)