# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:43:07 2023

@author: n.patsalidis
"""
import sys
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import numpy as np
import md_analysis as mda  
obj = mda.Analysis('init.gro','AAO.itp')
obj.read_file()
maxdist = 0.2
type1='Alt'
type2='O'
coordination = obj.calc_atomic_coordination(maxdist,type1,type2)
coordination.update(obj.calc_atomic_coordination(0.25,'Alo', 'O'))
t = coordination['Alt-O']
o = coordination['Alo-O']
ft = t==3
fo = o==6
xt = np.count_nonzero(ft)*100/t.shape[0]
xo =  np.count_nonzero(fo)*100/o.shape[0]
print('COORDINATION ACCURACY: Alt--> {:4.3f} %, Alo --> {:4.3f} %'.format(xt,xo))

ft_all = np.zeros(obj.natoms,dtype=bool)
for j,i in enumerate(np.where(obj.at_types=='Alt')[-1]):
    ft_all[i] =ft[j]
fo_all = np.zeros(obj.natoms,dtype=bool)
for j,i in enumerate(np.where(obj.at_types=='Alo')[-1]):
    fo_all[i] =fo[j]
obj.filter_the_system(ft_all)
obj.write_gro_file('alt.gro')
obj = mda.Analysis('init.gro','AAO.itp')
obj.read_file()
obj.filter_the_system(fo_all)
obj.write_gro_file('alo.gro')
obj = mda.Analysis('init.gro','AAO.itp')
obj.read_file()
mda.ass.clear_logs()