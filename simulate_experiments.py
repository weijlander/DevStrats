# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:15:33 2018

@author: Wouter Eijlander
"""

import Infant
import numpy as np
import tqdm
from saving import savedata
import math


n_infants = 10
cardinality = 5
n_babbles=5000
filename="data//results//"+str(cardinality)+"values"

#n_testing_targets=[144 for age in age_weeks]
age_weeks=[1,4,7,10,13,16,19]
n_testing_targets=[144 for age in age_weeks]
n_learning_reaches=[1000 for age in age_weeks]
cc_track=[0.0,0.0,0.8,0.6,0.3,0.2,0.2]

threshold_distance=10

all_extended_reaches=[]
all_reach_lengths=[]
all_reach_errors=[]
all_pred_errors=[]
all_axis_errors=[]

pbar=tqdm.tqdm(total=(n_infants*len(n_learning_reaches)*n_testing_targets[0]),desc="Processing "+str(n_infants)+" babies, current progress: ")

try:
    for inf_n in range(n_infants):
        # initialize a simulated infant, and the lists for monitored result statistics
        baby=Infant.Infant(card=cardinality)
        baby.motor_babbling(nb=n_babbles,max_cc=0.0)
        
        extended_reaches=[]
        reach_lengths=[]
        reach_errors=[]
        pred_errors=[]
        axis_errors=[]
        
        for cycle in range(len(n_learning_reaches)):
            # perform a learning cycle and a test cycle
            baby.motor_babbling(nb=n_learning_reaches[cycle],max_cc=cc_track[cycle])
            test_targets=baby.get_testtargets(n_testing_targets[cycle])
            ex_reaches=0
            
            for target in test_targets:
                # calculate reach error
                #nodes=list(baby.reach(target,cc_track[cycle])[0][1][:3])
                hand_pos,goal_axes,axes=baby.reach(target,cc_track[cycle])
                hand_pos=list(hand_pos[0][1][:3])
                reach_e=np.linalg.norm(np.subtract(hand_pos,target[0]))
                
                # calculate network prediction error
                pred_axes=axes
                pred_e=np.sum(abs(np.subtract(goal_axes,pred_axes))) 
                axis_e=tuple(abs(np.subtract(goal_axes,pred_axes)))
                
                # calculate reach length, and determine if it exceeds the threshold, and if so, add it to the extended reaches
                r_length=np.linalg.norm(hand_pos)
                ex_reaches+=(math.sqrt(math.pow(hand_pos[1],2)+math.pow(hand_pos[0],2))>threshold_distance)
                
                # add all the statistics to their respective lists
                reach_errors.append(reach_e)
                reach_lengths.append(r_length)
                pred_errors.append(pred_e)
                axis_errors.append(axis_e)
                pbar.update(1)
            extended_reaches.append(ex_reaches)
            
        # save this infant's results to the data matrices
        all_extended_reaches.append(extended_reaches)
        all_reach_lengths.append(reach_lengths)
        all_reach_errors.append(reach_errors)
        all_pred_errors.append(pred_errors)
        all_axis_errors.append(axis_errors)
    # save the data matrices to csv files
    savedata(filename,all_extended_reaches,all_reach_lengths,all_reach_errors,all_pred_errors,all_axis_errors)

except KeyboardInterrupt:
    # If I interrupt the run because it runs too long, or for some other reason, save the data that has been gathered so far (unless no data was gathered)
    if all_extended_reaches:
        savedata(filename,all_extended_reaches,all_reach_lengths,all_reach_errors,all_pred_errors,all_axis_errors)