# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""

import sys

from matplotlib import pyplot as plt

sys.path.insert(0,'\\Users\\n.patsalidis\\Desktop\\PHD\\REPOSITORIES\\MDanalysis')

import md_analysis as mda
import numpy as np


conftype='zdir'
dads = 1 ; 
phia=('C','C','CD','CD')
phib=('CD','C','C','CD')

trajf = '../trr/PRwhbench.trr'
traj = mda.Analysis_Confined(trajf,['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'],
                        conftype, 
                        topol_file ='../gro/alupb.gro',
                        memory_demanding=False,
                        particle='ALU',polymer='PB')
traj.read_file()

#my_distributionFilt ={'conformations': ['train','loop','tail','free']}
my_distributionFilt ={'space': [(0,1),(1,3),(3,5)]}

distr =traj.calc_dihedral_distribution(phib,
            dads=1.04,filters=my_distributionFilt)

colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']

fig =plt.figure(figsize=(3.5,3.5),dpi=300)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=5)
plt.tick_params(direction='in', which='major',length=10)
plt.xlim([-180,180])
plt.xlabel(r'$\phi$')
for i,(k,v) in enumerate(distr.items()):
    if type(k) is tuple:
        lab =r'$\in {}$ nm'.format(k)
    else:
        lab = k
    plt.hist(v,
        label=lab,histtype='step',ls='-',
        bins=300,density=True,color=colors[i])
plt.legend(frameon=False,fontsize=8)
plt.savefig('phi.png',bbox_inches='tight')
plt.show()
        

