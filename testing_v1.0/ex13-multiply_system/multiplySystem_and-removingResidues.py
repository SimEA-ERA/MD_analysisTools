# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""

import os
import sys

from matplotlib import pyplot as plt
import os
import numpy as np

sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import md_analysis as mda  



obj = mda.Analysis('confout.gro','topol_UA_PB30.itp')

obj.read_file()
multiplicity=(2,2,1) 
def fcrit (c): 
    fx = np.logical_or(c[:,0]<0.5,c[:,0]>19.5)
    fy = np.logical_or(c[:,1]<0.5,c[:,1]>19.5)
    fz = np.logical_or(c[:,2]<0.5,c[:,2]>30)
    return np.logical_or(fx,np.logical_or(fy,fz))
    
obj.multiply_periodic(multiplicity)
print(obj.box_mean())
obj.remove_residues(fcrit)
fname = 'mult_x{}y{}z{}-removedres.gro'.format(*multiplicity)
box = obj.get_box(0)
obj.timeframes[0]['boxsize']=np.array([20,20,30])
obj.write_gro_file(fname=fname)
mda.ass.try_beebbeeb()
mda.ass.clear_logs()