# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:15:11 2018

@author: Wouter Eijlander
"""

from move_arm import move_arm
from cost_f import cost_f

class ArmModel():
    def __init__(self,lengths=[[0.7,0.7,-9.2],[2.0,1.4,-18.4]],angles=[[10,10,10],[5,0,0]],limits=[[-20,130],[-20,70],[-70,60],[0,140]]):
        self.arm = lengths
        self.limits = limits
        self.angles = angles
        
    def move(self,muscles,coac):
        # performs a move and returns its outcome and an associated cost
        cost = cost_f(muscles,coac)
        return (move_arm(self.arm,muscles,self.limits,coac),cost)
    
    def angle_to_activation(self,angles):
        # determine what activation values should be used for given joint angles
        activations=[]
        # this works because we disregard the y and z components for the elbow, we don't use it, and they are the 5th an 6th angle
        for angle,limit in zip([a for joint in angles for a in joint],self.limits):
            activations.append((angle-limit[0])/(limit[1]-limit[0]))
        return activations