# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:24:35 2018

@author: Wouter Eijlander
"""

from unify_muscles import unify_muscles
from homog_transform import homog_transform
import numpy as np
import math

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
    # Determine Rotation matrices for both arm segments
    x1 = unify_muscles(muscles[0],muscles[1],coac)
    y1 = unify_muscles(muscles[2],muscles[3],coac)
    z1 = unify_muscles(muscles[4],muscles[5],coac)
    x2 = unify_muscles(muscles[6],muscles[7],coac)
    
    rx1 = limits[0][0]+(x1*(limits[0][1]-limits[0][0]))
    ry1 = limits[1][0]+(y1*(limits[1][1]-limits[1][0]))
    rz1 = limits[2][0]+(z1*(limits[2][1]-limits[2][0]))
    rx2 = limits[3][0]+(x2*(limits[3][1]-limits[3][0]))
    
    return (calc_arm(arm,[rx1,ry1,rz1],[rx2,0,0]),[x1,y1,z1,x2])#list(np.zeros(np.shape(rx2))),list(np.zeros(np.shape(rx2)))])

def calc_arm(arm,shoulder,elbow):
    '''
    Calculate the new end-effectors for the given arm segments given a rotation for both joints
    @param shoulder: rotation vector for the shoulder
    @type shoulder: [rx,ry,rz]
    @param elbow: rotation vector for the elbow. In reality, only the first value is used since the elbow has only the x-axis DOF
    @type elbow: [rx,ry,rz]
    @param arm: end-effector positions of the segments of the arm
    @type arm: [[x,y,z],[x,y,z]]
    '''
    # determine positional relationships between segments, needed for correctly
    # calculating forearm movement
    [l1,l2] = arm
    d2 = np.subtract(l2,l1)
    
    rx1,ry1,rz1 = shoulder
    rx2,ry2,rz2 = elbow
    
    # calculate end-effector position
    e1 = homog_transform(l1[0],l1[1],l1[2],0,0,0,rx1,ry1,rz1) # Perform the rotation to the upper arm
    #e2 = homog_transform(d2[0],d2[1],d2[2],e1[0],e1[1],e1[2],0,0,rz1) # determine the new forearm position based on upper arm translation and rotation
    e3 = homog_transform(d2[0],d2[1],d2[2],0,0,0,rx1*abs(rx2-140),ry1,rz1) # perform forearm rotation
    return [e1,np.add(e3,e1)]

def clamp_degrees(angles,limits=[[-20,130],[-20,70],[-70,60],[0,140],[0,0],[0,0]]):
    i=0
    clamped_angles=list()
    rounded_angles=round_angles(angles)
    for joint in rounded_angles:
        n_angles=[]
        for angle in joint:
            clamped_angle=max(angle,limits[i][0])
            clamped_angle=min(clamped_angle,limits[i][1])
            n_angles.append(clamped_angle)
            i+=1
        clamped_angles.append(n_angles)
    return clamped_angles

def inverse_approx(tar,arm,angles=[[1,1,1],[1,0,0]],h=0.3,eps=1,maxit=1000):
    '''
    @param tar: target position in 3D
    @type tar: [x,y,z]
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
    diff=np.linalg.norm(np.subtract(tar,arm[-1][:3]))
    best = ([],50.0)
    while diff>eps and it < maxit:
        # get the current iteration's change in joint angles
        dO=get_delta(tar,[l[:3] for l in arm],angles)
        # change the known angles by adding the weighted change in angles
        angles=clamp_degrees(list(np.add(angles,np.multiply(dO,h))))
        # recalculate the position of the arm after the angle change is applied
        arm=calc_arm(arm,angles[0],angles[1])
        it+=1
        diff = np.linalg.norm(np.subtract(tar,arm[-1][:3]))
        best=(angles,diff) if diff<best[1] else best
    return round_angles(best[0]),best[1]

def get_delta(tar,arm,angles):
    '''
    @param tar: target position in 3d
    @type tar: [x,y,z]
    @param arm: end-effector positions o the segments of the arm
    @type arm: [[x,y,z],[x,y,z]]
    @param angles: the angles in 3d for both joints. Only used for calculatng the jacobian
    @type angles: [[dxsh,dysh,dzsh],[dxel,dyel,dzel]]
    '''
    #jac_t = get_damped_ls(tar,arm,angles,0.02)
    jac_t = get_transJac(tar,arm,angles)
    diff = list(np.subtract(tar,arm))
    dO = list(np.multiply(jac_t,diff))
    return dO

def get_damped_ls(tar,arm,angles,d):
    jac_t=get_transJac(tar,arm,angles)
    jac=np.ndarray.tolist(np.transpose(jac_t))
    #jac_t*(jac*jac_t+math.pow(d,2)*i)^-1
    i=np.identity(len(jac))
    a=np.dot(jac,jac_t)
    b=math.pow(d,2)*i
    c=np.linalg.inv(a+b)
    damped=np.ndarray.tolist(np.dot(jac_t,c))
    return damped

def get_transJac(tar,arm,angles):
    l1,l2=arm
    shoulder,elbow=angles
    jac_t=list()
    
    #calculate the jacobian entries for both joints
    j_l1=list(np.cross(shoulder,np.subtract(l2,[0,0,0])))
    j_l2=list(np.cross(elbow,np.subtract(l2,l1)))
    
    jac_t.append(j_l1)
    jac_t.append(j_l2)
    return jac_t

def round_angles(angles):
    fixed = np.ndarray.tolist(np.zeros(np.shape(angles)))
    for row in range(len(angles)):
        for angle in range(len(angles[angles==row])):
            fixed[row][angle] = (angles[row][angle]%360.0) if angles[row][angle]>0.0 else -(abs(angles[row][angle])%360.0)
    return fixed