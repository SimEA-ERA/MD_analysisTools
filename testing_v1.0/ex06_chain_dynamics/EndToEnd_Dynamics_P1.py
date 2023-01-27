"""
Created on Sat Jun 25 22:37:42 2022

@author: n.patsalidis
"""
import sys
from matplotlib import pyplot as plt
import matplotlib

from time import perf_counter
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')

t0 = perf_counter()
import md_analysis as mda  
import numpy as np
##### Setup ####

conftype = 'zdir'
dads=1.025
trajf='../trr/PRwh_dt1.trr'


results = mda.Analysis_Confined(trajf,
                    ['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'],
                    conftype, 
                    topol_file ='../gro/alupb.gro',
                    particle='ALU',polymer='PB')

results.read_file()
#calculation 
ree_t, fRee_t = results.calc_Ree_t()
ReeDy = results.Dynamics('P1',ree_t)
#######################

size = 3.5
################
#Plots #########
figsize=(size,size)
dpi = 300
fig= plt.figure(figsize=figsize,dpi=dpi)
plt.xscale('log') 
x = mda.ass.numpy_keys(ReeDy)/1000
y= mda.ass.numpy_values(ReeDy)
x = x[1:]
y = y[1:]
plt.plot(x,y,lw = size/2)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=1.5*size)
plt.tick_params(direction='in', which='major',length=3*size)
plt.xlabel(r'$t$ $(ns)$',fontsize=3*size)
plt.xticks(fontsize=3*size)
plt.yticks(fontsize=3*size)
plt.ylabel(r'$P_1(t)$',fontsize=3*size)
plt.legend(frameon=False,fontsize=2.2*size)
plt.savefig('P1_endtoend.png',bbox_inches='tight')
plt.show()


mda.ass.print_time(perf_counter()-t0, 'MAIN',1000)
mda.ass.try_beebbeeb()
mda.ass.clear_logs()