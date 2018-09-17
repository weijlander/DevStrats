# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:24:35 2018

@author: Wouter Eijlander
"""

from unify_muscles import unify_muscles
from homog_transform import homog_transform
import numpy as np

def move_arm(arm,muscles,limits,coac):
    '''
    arm:
        list[segments] containing segments:
            list[float] containing 3 floats indicating the endpoint for a segment
    muscles:
        list[floats] containing 8 muscle activations
    limits:
        list[range] containing 4 rotation ranges for each DOF:
            list[float] containing the min and max rotation value for that DOF
    coac:
        float (0:1) coactivation coefficient
    '''
    [l1,l2] = arm
    
    # Determine Rotation matrices for both arm segments
    x1 = unify_muscles(muscles[0],muscles[1],coac)
    y1 = unify_muscles(muscles[2],muscles[3],coac)
    z1 = unify_muscles(muscles[4],muscles[5],coac)
    x2 = unify_muscles(muscles[6],muscles[7],coac)
    
    rx1 = limits[0][0]+(x1*(limits[0][1]-limits[0][0]))
    ry1 = limits[1][0]+(y1*(limits[1][1]-limits[1][0]))
    rz1 = limits[2][0]+(z1*(limits[2][1]-limits[2][0]))
    rx2 = limits[3][0]+(x2*(limits[3][1]-limits[3][0]))
    
    # determine positional relationships between segments, needed for correctly
    # calculating forearm movement
    d2 = np.subtract(l2,l1)
    
    # calculate end-effector position
    e1 = homog_transform(l1[0],l1[1],l1[2],0,0,0,rx1,ry1,rz1) # Perform the rotation to the upper arm
    e2 = homog_transform(d2[0],d2[1],d2[2],e1[0],e1[1],e1[2],0,0,rz1) # determine the new forearm position based on upper arm translation and rotation
    e3 = homog_transform(e2[0],e2[1],e2[2],0,0,0,rx2,0,0) # perform forearm rotation
    return (e1,e3)