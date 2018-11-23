# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 10:50:26 2018

@author: Wouter Eijlander
"""

from ppt import pyPPTwrapper
from ppt.CPTable import CPTable
from ppt.DSL_network import DSL_network
from ppt.EDistr import EDistr
from ppt.HPrior import HPrior
from ppt.PDistr import PDistr
from ppt.PPToolbox import PPToolbox
import numpy as np
from nodes import *

class armNet():
    def __init__(self,name='def',card=10):
        self.name=name
        #self.posits=[Node(i,[]) for i in ['X','Y','Z']]
        self.axes=[Axis(i,[],card=card) for i in ['shx','shy','shz','elx']]
        self.muscles=list()
        for a in self.axes:
            self.muscles.append(Musc(a.label+'_ag',[a],card=card))
            self.muscles.append(Musc(a.label+'_ant',[a],card=card))
        self.cc=[Coeff('CC',self.axes,card=card)]
        self.nodes={node.label:node for sublist in [self.axes, self.muscles, self.cc] for node in sublist}
        for musc in self.muscles:
            musc.extend_hp([self.nodes[n] for n in self.nodes])
        
    def reset(self):
        for node in self.nodes:
            node.value=0.0