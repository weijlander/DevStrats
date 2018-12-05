# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:04:34 2018

@author: Wouter Eijlander
"""

import numpy as np
import scipy.stats as stats
import itertools
import tqdm
import math
from ArmModel import ArmModel
from t_distr import t_distr
from unify_muscles import unify_muscles
from worlds import World
from move_arm import *
from ArmNet import *

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
        self.worlds=[]
    
    def reach(self,target,cc):
        '''
        @param target: the target position in 3d and its width
        @type target: tuple(list[x,y,z],float)
        @param cc: the given coactivation coefficient
        @type cc: float
        '''
        # Determine the target position that the agent sees (this can differ from the actual target centre)
        target_distr = t_distr([target],self.drange)
        sampled_points = np.ndarray.tolist(np.amax(target_distr,axis=1))
        (x,y,z) = target_distr[0].index(sampled_points[0]),target_distr[1].index(sampled_points[1]),target_distr[2].index(sampled_points[2])
        target_pos = (self.drange[1][x],self.drange[1][y],self.drange[1][z])
        
        # Determine the values for all the nodes in the agent's arm network by inference
        nodes,goal_axes=self.infer_nodes(target_pos,cc)
        
        # Update beliefs in the network
        self.update_hparams([n for n in self.anet.nodes],nodes)
        for n in self.anet.nodes:
            self.anet.nodes[n].update_pd()
        for subnet in self.worlds:
            for world in subnet:
                world.update_world(self.anet)
        
        result=self.rArm.move(nodes[4:-1],cc)
        end_eff=result[0]
        axes=end_eff[1]
        return (end_eff,goal_axes,axes)
    
    def motor_babbling(self,nb=1000,type='gaussian',width=0.5,max_cc=0.0):
        '''
        build a knowledge base for the probability distributions over network's nodes
        @param nb: the number of random movements
        @type nb: int
        @param type: the type of distributions from which the leaf nodes in the network will be sampled: 'gaussian' or 'uniform', defaults to gaussian
        @type type: string
        '''
        #for cycle in tqdm.tqdm(range(nb),desc='Motor babbling:'):
        for cycle in range(nb):
            values=[]
            # sample random muscle activations based on the babbling type
            for muscle in self.anet.muscles:
                # sample a random muscle activation and clip
                ranm=np.random.normal(loc=0.5,scale=width)
                ranm=min(max(ranm,0),max_cc)
                values.append(ranm)
            # sample a random cc and clip
            rancc = np.random.normal(loc=0.2,scale=width)
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
            self.update_hparams([n for n in self.anet.nodes],nodes)
        for n in self.anet.nodes:
            self.anet.nodes[n].update_pd()
        self.worlds=np.ndarray.tolist(np.transpose(self.get_worlds()))
    
    def infer_nodes(self,target,cc):
        '''
        Infer the most probable values for the network's nodes given the target and the predecided cc
        @param target: the target to reach toward
        @type target: tuple(list[x,y,z],float)
        @param cc: the coactivation coefficient
        @type cc: float
        '''
        angles=inverse_approx(target,self.rArm.arm,angles=self.rArm.angles)[0]
        axes=self.rArm.angle_to_activation(angles)
        
        cc=self.anet.nodes['CC'].get_bin(cc)
        fixed_axes=[self.anet.nodes[a].get_bin(axes[i]) for i,a in enumerate(['shx','shy','shz','elx'])]
        
        nodes=[[],[]]
        for i,axis in enumerate(self.worlds):
            after_cc = self.shift_mass(axis,'CC',cc)
            
            ax=fixed_axes[i]
            after_axis = self.shift_mass(after_cc,['shx','shy','shz','elx'][i],ax)
            
            ag = self.decide_value(after_axis,['shx','shy','shz','elx'][i]+'_ag')
            after_ag = self.shift_mass(after_axis,['shx','shy','shz','elx'][i]+'_ag',ag)
            
            # this might not be fully necessary! the remaining worlds now should only be the 10 different values of ant, from which we can just pick the best
            ant = self.decide_value(after_ag,['shx','shy','shz','elx'][i]+'_ant')
            after_ant = self.shift_mass(after_ag,['shx','shy','shz','elx'][i]+'_ant',ant)
            
            # add the nodes to what we know
            nodes[0].extend([after_ant[0].labels])
            nodes[1].extend([after_ant[0].values])
        
        # turn the node labels and values into a dictionary to remove duplicates (CC is the cuplrit here)
        solution = {lab:nodes[1][i][j] for i,node in enumerate(nodes[0]) for j,lab in enumerate(node)}
        values=[solution[l] for l in self.anet.nodes]
            
        return (values,fixed_axes)
    
    def decide_value(self,worlds,label,mode='softmax'):
        '''
        Decide the value to take for the variable associated with given label, based on the worlds we have remaining
        TODO: currently just takes the max, perhaps perform some simulated annealing instead
        @param worlds: the remaining worlds to decide the best subset from
        @type worlds: list[World]
        @param label: the label for the variable that we're going to decide on now
        @type label: String
        '''
        
        if mode=='max':
            ordered_list=[w.probmass+w.added_mass for w in worlds]
            chosen_i=ordered_list.index(np.amax(ordered_list))
        elif mode=='softmax':
            ordered_list=[w.probmass+w.added_mass for w in worlds]
            chosen_i=self.softmax(ordered_list)
        elif mode=='entropy':
            chosen_i=self.choose_by_entropy(worlds,label)
        
        vals,labs=(worlds[chosen_i].values,worlds[chosen_i].labels)
        return vals[labs.index(label)]
    
    def choose_by_entropy(self,worlds,label):
        '''
        this one is a doozy: from the current worlds, choose the most entropic value for the variable with the given label 
        '''
        ordered_list=[w.probmass+w.added_mass for w in worlds]
        chosen_i=ordered_list.index(np.amax(ordered_list))
        for world in worlds:
            index=chosen_i
        return index
    
    def softmax(self,values):
        # sample a value from the given probdistr by their probability
        chosen_value=np.random.choice(values,p=values)
        # and if multiple instances of that value exit, then get all indices...
        indices=[i for i,x in enumerate(values) if x==chosen_value]
        # and choose one at random, since they're all equally likely
        index = indices[round(np.random.uniform(low=0,high=len(indices)-1))]
        return index
    
    def shift_mass(self,worlds,label,value):
        '''
        Shift the probability mass from all worlds that don't satisfy node(label)==value to all worlds that do satisfy this condition
        @param worlds: the worlds that we still have on consideration
        @type worlds: list[World]
        @param label: the label of the variable that we now image over
        @type label: string
        @param value: the value of the variable that we now image over
        @type value: float
        '''
        remaining = []
        eliminated = []
        for world in worlds:
            if world.values[world.labels.index(label)]==value:
                remaining.append(world)
            else:
                eliminated.append(world)
        added_masses=np.ndarray.tolist(np.zeros(np.shape(remaining)))
        for elim in eliminated:
            added_masses=np.ndarray.tolist(np.add(elim.spread_probability_mass(remaining),added_masses))
        for i,rem in enumerate(remaining):
            rem.added_mass+=added_masses[i]
        return remaining
    
    def get_worlds(self):
        labels=[l for l in self.anet.nodes]
        subnets=[['shx','shx_ag','shx_ant','CC'],['shy','shy_ag','shy_ant','CC'],['shz','shz_ag','shz_ant','CC'],['elx','elx_ag','elx_ant','CC']]
        nworlds=pow(len(self.anet.nodes['shx'].values),len(subnets[0]))
        vals=[self.anet.nodes['shx'].values for node in subnets[0]]
        values=list(itertools.product(*vals))
        
        worlds=[]
        for wn in range(nworlds):
            #determine world i's value set, and make the world
            v=list(values[wn])
            d_sep_subnets = [World(self.anet,labs,v) for labs in subnets]
            worlds.append(d_sep_subnets)
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
            
    def get_testtargets(self,n):
        '''
        return a list of n targets that are just out of reach.
        Note: based on von Hofsten (1984), the targets are generated close to z-values of 0, in line with the simulated infant's eyeheight.
        '''
        testset=[]
        while len(testset)<n:
            a = -30
            b = 30
            x = (b-a)*np.random.uniform()+a
            a = 10
            b = 25
            y = (b-a)*np.random.normal(loc=0.5,scale=0.2)+a
            distance = float(int(math.sqrt(math.pow(x,2)+math.pow(y,2))))
            z=np.random.normal(loc=0.0,scale=0.5)
            if distance<25 and distance>21 and y>12:
                testset.append(([x,y,z],3))
            
        return testset
        