# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""
import sys
from matplotlib import pyplot as plt

sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import md_analysis as mda  
import numpy as np
##### Setup ####

conftype = 'zdir'
binl=0.005 ; dmax = 1

trajf = '../trr/PRwhbench.trr'
results = mda.Analysis_Confined(trajf,['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'],
                    conftype, 
                    topol_file ='../gro/alupb.gro',
                    memory_demanding=False,
                    particle='ALU',polymer='PB')

results.read_file()
#finds pair distribution of type Alt and O
pd_AltO = results.calc_pair_distribution(binl,dmax,'Alo','O',density='coordination')

figsize = (3.3,3.3)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=5)
plt.tick_params(direction='in', which='major',length=10)
plt.xlim([0,dmax+binl])
#plt.yscale('log') ; 
plt.xlabel(r'$r (nm)$')
plt.ylabel(r' < O atoms in distance from Alt >')
plt.plot(pd_AltO['d'],pd_AltO['gr'],marker='o',fillstyle='none',label = r'$Alo-O$',color='green')
#plt.plot(pd_all['d'],pd_all['gr'],marker='o',ls='--',lw=0.5,fillstyle='none',label = r'$all-all',color='k')
plt.legend(frameon=False,fontsize=8)
plt.savefig('Alo-O.png',bbox_inches='tight')
plt.show()



    