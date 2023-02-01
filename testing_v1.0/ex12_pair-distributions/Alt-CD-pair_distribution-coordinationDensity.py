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
binl=0.01 ; dmax = 1.5

trajf = '../trr/PRwhbench.trr'
results = mda.Analysis_Confined(trajf,['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'],
                    conftype, 
                    topol_file ='../gro/alupb.gro',
                    memory_demanding=False,
                    particle='ALU',polymer='PB')

results.read_file()
#finds pair distribution of type CD and Alt
pd_AltCD = results.calc_pair_distribution(binl,dmax,'Alt','CD',density='number')
# finds pair distribution of type Alt and Alt
pd_AltAlt = results.calc_pair_distribution(binl,3,'Alt','Alt',density='number')
# finds pair distribution of type Alt and all atoms
pd_Alt = results.calc_pair_distribution(binl,dmax,'Alt',density='number')
# all atoms to all (number of Alt is about 52 times less than system number of atoms.It takes an order of magnitute more time from previous one)
#pd_all = results.calc_pair_distribution(binl, dmax)

figsize = (3.3,3.3)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=5)
plt.tick_params(direction='in', which='major',length=10)
plt.xlim([0,dmax+binl])
#plt.ylim([0,2])
plt.xlabel(r'$r (nm)$')
plt.ylabel(r'<# density of type 2 atoms > \ $nm^{-3}$')
plt.plot(pd_AltCD['d'],pd_AltCD['gr'],
         marker='o',fillstyle='none',label = r'$Alt-CH$=',color='green')
plt.plot(pd_AltAlt['d'],pd_AltAlt['gr'],lw=0.6,ls='--',
         marker='o',fillstyle='none',label = r'$Alt-Alt$',color='red')
#plt.plot(pd_Alt['d'],pd_Alt['gr'],marker='o',ls='--',lw=0.5,fillstyle='none',label = r'$Alt-all$',color='blue')
#plt.plot(pd_all['d'],pd_all['gr'],marker='o',ls='--',lw=0.5,fillstyle='none',label = r'$all-all',color='k')
plt.legend(frameon=False,fontsize=8)
plt.savefig('Alt-CD_Alt-Alt-coordinationDensity.png',bbox_inches='tight')
plt.show()



    