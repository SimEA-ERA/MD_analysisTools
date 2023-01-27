# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""
import sys
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import md_analysis as mda  

import numpy as np
##### Setup ####

trajf = '../trr/PRwh_dt1.trr' # trajectory file
conftype = 'zdir' #type of confinmnent
connectivity_info = ['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'] # bond information

obj = mda.Analysis_Confined(trajf, #trajectory file
        connectivity_info, # can be either a list of files or just one file
        conftype, # signifies what functions to use to calculate e.g. the distance, or the volume of each bin
        topol_file ='../gro/alupb.gro', # if it's gromacs setup we need a gro file of one frame to read atom types, molecule types and exetra 
        particle='ALU',polymer='PB') # Need to give the particle and polymer name 

#obj.read_file()
binl=0.025 ; dmax =4.825 

# if you dont read the file the algorithm will read it 
# automatically for you when you try to calculate any property

dp = obj.calc_density_profile(binl,dmax,flux=True) # returns a dictionary containing the data in arrays
# For plotting consult matplotlib, matplotlib.pyplot documentation
from matplotlib import pyplot as plt
size=3.3
figsize=(size,size)
dpi=300
fig,host = plt.subplots(dpi=dpi,figsize=figsize)
par1 = host.twinx()
caxis='green'
host.minorticks_on()
host.tick_params(direction='in', which='minor',length=5)
host.tick_params(direction='in', which='major',length=10)
host.set_ylabel(r'$\rho_{flux}(g/cm^3)$')
host.set_xlabel(r'$d(nm)$')
par1.tick_params(axis='y',colors=caxis)
par1.spines["right"].set_edgecolor(caxis)
par1.set_ylabel(r'$\rho(g/cm^3)$',color=caxis)
plt.xlim([0,5])
par1.plot(dp['d'],dp['rho'],label=r'$\rho_{tot}$',color=caxis)
par1.legend(frameon=False,loc='upper right')
host.plot(dp['d'],dp['rho_flux']**0.5,label= r'$\rho_{flux}$',
          marker='o',ls='-',lw=0.0,color='k',fillstyle='none',markersize=2.5)  
host.legend(frameon=False,loc = 'upper center')
plt.savefig('mass_density_std.png',bbox_inches='tight')
plt.show()