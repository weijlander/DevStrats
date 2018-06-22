# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 12:55:33 2018

@author: Wouter Eijlander
"""

import numpy as np
import math

def  homog_transform(x,y,z,tx,ty,tz,dx,dy,dz):
    T = [1,0,0,tx,0,1,0,ty,0,0,1,tz,0,0,0,1]
    Rx = [1,0,0,0,0,math.cos(math.radians(dx)),-math.sin(math.radians(dx)),0,0,math.sin(math.radians(dx)),math.cos(math.radians(dx)),0,0,0,0,1]
    Ry = [math.cos(math.radians(dy)), 0, math.sin(math.radians(dy)), 0, 0, 1, 0, 0, -math.sin(math.radians(dy)), 0, math.cos(math.radians(dy)), 0, 0, 0, 0, 1]
    Rz = [math.cos(math.radians(dz)), -math.sin(math.radians(dz)), 0, 0, math.sin(math.radians(dz)), math.cos(math.radians(dz)), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    
    T = np.reshape(T,(4,4))
    Rx = np.reshape(Rx,(4,4))
    Ry = np.reshape(Ry,(4,4))
    Rz = np.reshape(Rz,(4,4))
    xyz = np.array([x,y,z,1])
    
    H = np.dot(np.dot(np.dot(np.dot(T,Rx),Ry),Rz),xyz)
    return H