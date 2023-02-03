# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:43:07 2023

@author: n.patsalidis
"""
import sys
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import numpy as np
import md_analysis as mda  
obj = mda.Analysis('cube_mod.gro','../itp/alu_to.itp')
obj.read_file()
maxdist = 0.2
type1='Alt'
type2='O'
coordination_cube = obj.calc_atomic_coordination(maxdist,type1,type2)
coordination_cube.update(obj.calc_atomic_coordination(0.28,['Alo','Alt'], 'O'))
t = coordination_sub['Alt-O']
o = coordination_sub['Alo-O']
xt = np.count_nonzero(t==3)*100/t.shape[0]
xo =  np.count_nonzero(o==6)*100/o.shape[0]
print('COORDINATION ACCURACY: Alt--> {:4.3f} %, Alo --> {:4.3f} %'.format(xt,xo))
mda.ass.clear_logs()