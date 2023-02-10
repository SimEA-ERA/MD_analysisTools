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
topol_vector = ['C','CD','CD','C']
trajf = '../trr/PRwh_dt1.trr'
################

DK= dict()
obj = mda.Analysis_Confined(trajf,
                ['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'],
                conftype, 
                topol_file ='../gro/alupb.gro',
                particle='ALU',polymer='PB')

obj.read_file()

ads_seg_t = obj.calc_adsorbed_segments_t(topol_vector,dads)
    
DK['seg'] = obj.Kinetics(ads_seg_t)
#When we call kinetics we only need boolean data


du, fwads = obj.calc_chainCM_t(filters={'adsorption':None},dads=dads)
#we also give the degree of adsorption as weights. It will perform a weighted average
DK['chains'] = obj.Kinetics(fwads['ads'])

## GO GO PLOT ## 
size = 5.5

figsize = (size,size)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=size*1.5)
plt.tick_params(direction='in', which='major',length=size*3)
plt.xscale('log')
plt.xticks(fontsize=2.5*size)
plt.yticks(fontsize=2.5*size)
plt.ylabel(r'$DK(t)$ ',fontsize=size*3)
plt.xlabel(r'$t$ $(ns)$',fontsize=size*3)
for i,(k,dk) in enumerate(DK.items()):
    x = mda.ass.numpy_keys(dk)/1000
    y = mda.ass.numpy_values(dk)
    plt.plot(x,y,ls='none',marker = 'o',
         markersize=size,fillstyle='none',label=k)
plt.legend(frameon=False,ncol=1,fontsize=size*2.5)
plt.savefig('DK.png',bbox_inches='tight')
plt.show()


mda.ass.print_time(perf_counter()-t0, 'MAIN',1000)
mda.ass.try_beebbeeb()
