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

trajf = '../trr/PRwhbench.trr' # trajectory file
conftype = 'zdir' #type of confinmnent
connectivity_info = ['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'] # bond information

obj = mda.Analysis_Confined(trajf, #trajectory file
        connectivity_info, # can be either a list of files or just one file
        conftype, # signifies what functions to use to calculate e.g. the distance, or the volume of each bin
        topol_file ='../gro/alupb.gro', # if it's gromacs setup we need a gro file of one frame to read atom types, molecule types and exetra 
        particle='ALU',polymer='PB') # Need to give the particle and polymer name 

#obj.read_file()
binl=0.025 ; dmax =4.825 ; dads = 1.025 

# if you dont read the file the algorithm will read it 
# automatically for you when you try to calculate any property

conf_dp = obj.calc_density_profile(binl,dmax,dads=dads,
                                        option='conformations') # returns a dictionary containing the data in arrays

#use mda assistant class to print your conformational statistics
mda.ass.print_stats(conf_dp['stats'])
# For plotting consult matplotlib, matplotlib.pyplot documentation
    

from matplotlib import pyplot as plt 

fig =plt.figure(figsize=(3,3),dpi=300)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=5)
plt.tick_params(direction='in', which='major',length=10)
plt.xlabel(r'$d(nm)$')
plt.ylabel(r'$\rho(g/cm^3)$')
plt.xlim([0,5])


plt.plot(conf_dp['d'],conf_dp['mrho'],label=r'$\rho_{tot}$',color='green')
plt.plot(conf_dp['d'],conf_dp['mloop'],label='loop',ls='--',lw=1.5,color='red')
plt.plot(conf_dp['d'],conf_dp['mtail'],label='tail',ls='--',lw=1.5,color='blue')
plt.plot(conf_dp['d'],conf_dp['mfree'],label='free',ls='--',lw=2,color='orange')
   
plt.legend(frameon=False)
plt.savefig('mass_density_confs.png',bbox_inches='tight')
plt.show()
    
    

mda.ass.clear_logs()