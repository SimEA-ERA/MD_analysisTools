# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:43:07 2023

@author: n.patsalidis
"""
import sys
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import numpy as np
import md_analysis as mda  
obj = mda.Analysis_Confined('cube.gro','../itp/alu_to.itp',
                            'zdir',particle='ALU',polymer='ALU')
obj.read_file()
maxdist = 0.2

c = obj.get_coords(0)

cmin = c.min(axis=0)[2] + maxdist
cmax = c.max(axis=0)[2] - maxdist

counter = 0

for i in range(c.shape[0]):
    zi = c[i,2]
    if obj.at_types[i] == 'Alt':
        if cmin < zi < cmax:
            counter+=1    
            obj.at_types[i] == 'Alo'
obj.write_gro_file('cube_mod.gro')
print(counter)

