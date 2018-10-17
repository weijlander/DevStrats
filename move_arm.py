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
    @param arm: list of 3D end-points for each segment of the arm
    @type arm: list[segments]
        @type segments: list[floats] 3 end-effector float positions
    @param muscles: list containing 8 muscle activations
    @type muscles: list[float (0:1)]
    @param limits: rotation limits for all the joint axes
    @type limits: list[range] containing 4 rotation ranges for each DOF
        @typee range: list[float] containing the min and max rotation value for that DOF
    @param coac: coactivation coefficient
    @type coac: float (0:1) 
    @return endpoints for the upper arm and forearm
    @type return tuple(list)
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

def inverse_approx(tar,arm,angles=[[0,0,0],[0,0,0]],h=0.001,eps=0.8,maxit=30):
    '''
    @param tar: target position in 3D
    @type tar: list(x,y,z) where x,y,z,=float
    @param arm: end-effector positions of the segments of the arm
    @type arm: [[x,y,z],[x,y,z]]
    @param angles: starting angles for both joints
    @type angles: list(dsx,dsy,dsz,dex)Assumes start position arm is at joint angles 0 degrees.
    @param eps: epsilon value of desired minimum distance
    @type eps: float
    @return O: orientations for both joints
    @type O: [[x,y,z],[x,y,z]]
    '''
    it = 0
    while np.linalg.norm(np.subtract(tar,arm[-1]))>eps and it < maxit:
        dO=getDelta()
        angles+=list(np.multiply(dO,h))
        it+=1
        arm='bla' # need to re-calculate arm as in lines 42-47. Make this into a function that takes arm and rotations, and returns arm.
    return angles

def getDelta(tar,):
    
    return dO