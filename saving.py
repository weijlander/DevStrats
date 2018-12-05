# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:15:59 2018

@author: Wouter Eijlander
"""

import csv

def savedata(fn,axr,arl,are,ape,aae):
    with open(fn+'_extreaches.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(axr)
    with open(fn+'_reachlengths.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(arl)
    with open(fn+'_reacherrors.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(are)
    with open(fn+'_prederrors.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(ape)
    with open(fn+'_axiserrors.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(aae)