# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:15:11 2018

@author: Wouter Eijlander
"""

from move_arm import move_arm
from cost_f import cost_f

class ArmModel():
    def __init__(self,lengths=[[0,0,-10],[0,0,-20]],limits=[[-20,130],[-20,70],[-70,60],[0,140]]):
        self.arm = lengths
        self.limits = limits
        
    def move(self,muscles,coac):
        cost = cost_f(muscles,coac)
        return (move_arm(self.arm,muscles,self.limits,coac),cost)