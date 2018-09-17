# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:51:33 2018

@author: Wouter Eijlander
"""

from homog_transform import homog_transform
from unify_muscles import unify_muscles

def move_eyes(vis,muscles,limits,coac):
    '''
    vis:
        list[float] containing 3 floats indicating the current focal point in x,y,z respectively
    muscles:
        list[floats] containing 5 muscle activations
    limits:
        list[range] containing 3 rotation ranges for each DOF:
            list[float] containing the min and max rotation value for that DOF
    coac:
        float (0:1) coactivation coefficient
    '''
    # turns the eye around the x- and z-axes (upward and sideward respectively)
    agx = max(muscles[:2])
    antx = min(muscles[:2])
    agz = max(muscles[2:4])
    antz = min(muscles[2:4])
    
    # x and z are treated as antagonistic muscles- y is treated as a
    # higher-level depth fixation value
    x = unify_muscles(agx,antx,coac)
    y = muscles[4]
    z = unify_muscles(agz,antz,coac)
    
    # x and z movement are rotations along axes, y movement is depth fixation
    # and is thus a change to vector length
    rx = limits[0][0]+(x*(limits[0][1]-limits[0][0]))
    ty = limits[1][0]+(y*(limits[1][1]-limits[1][0]))
    m = ty/vis[1]
    vis = [v*m for v in vis]
    rz = limits[2][0]+(z*(limits[2][1]-limits[2][0]))
    
    return homog_transform(vis[0],vis[1],vis[2],0,0,0,rx,0,rz)