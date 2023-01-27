# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 22:37:42 2022

@author: n.patsalidis
"""
import sys
from matplotlib import pyplot as plt
import matplotlib

sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')

import md_analysis as mda  
import numpy as np


##### Setup ####



trajf = '../trr/PRwh_dt1.trr'
conftype = 'zdir'
binl=0.025 ; dmax =4.5 ; dads=1.025
phia=('C','C','CD','CD')
phib=('CD','C','C','CD')
dads = 1.025
layers = [(dads,1+dads),(dads+1,dmax)]
conformations = ['train','loop','tail','free']
PR = mda.Analysis_Confined(trajf,
                           ['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'],
                        conftype,
                        topol_file = '../gro/alupb.gro',
                        particle='ALU',polymer='PB')
PR.read_file()

B = mda.Analysis_Confined('../trr/Bwh_dt1.trr',
                          ['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'],
                conftype, 
                topol_file ='../gro/alupb.gro',
                particle='ALU',polymer='PB')
B.read_file()

B.append_timeframes(PR) # you can append the timeframes of an object. To have a meaning must have the same topology

#computation
confs_t = B.calc_conformations_t(dads)
  



nt = {k:v for k,v in confs_t.items() if 'n_' in k} #take the absolute numbers of the computed data
xt = {k:v for k,v in confs_t.items() if 'x_' in k} #take the percentages of the computed data 


#Plotting
figsize=(3.5,3.5) ; dpi=300
fig= plt.figure(figsize=figsize,dpi=dpi) 
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=5)
plt.tick_params(direction='in', which='major',length=10)
plt.xlabel(r'$t$ $(ns)$',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(r'number of atoms',fontsize=14)
i = 0
for k,v in nt.items():
    if 'bridge' in k or 'ads_chains' in k: continue
    x = np.array( list(v.keys()))/1000
    y = np.array( list(v.values()) )
    if 'train' in k: ntrain = y[-1000:].mean()
    plt.plot(x[::20],y[::20],  ls = 'none',marker='o',
             fillstyle='none',markersize=3,label=k.split('_')[1])
    i+=1
plt.legend(frameon=False,ncol=2,fontsize=12)
plt.savefig('nt.png',bbox_inches='tight')
plt.show() 

fig= plt.figure(figsize=figsize,dpi=dpi) 
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=5)
plt.tick_params(direction='in', which='major',length=10)
plt.xlabel(r'$t$ $(ns)$',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(r'conformation fraction (%)',fontsize=14)
i = 0
for k,v in xt.items():
    if 'bridge' in k or 'ads_chains' in k: continue
    x = np.array( list(v.keys()))/1000
    y = np.array( list(v.values()) )*100
    plt.plot(x[::20],y[::20],  ls = 'none',marker='o',
             fillstyle='none',markersize=3,label=k.split('_')[1])
    i+=1
plt.legend(frameon=False,ncol=2,fontsize=12)
plt.savefig('xt.png',bbox_inches='tight')
plt.show() 


 
fig= plt.figure(figsize=figsize,dpi=dpi)
#plt.xscale('log') 

plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=5)
plt.tick_params(direction='in',axis='y', which='minor',length=0)
plt.tick_params(direction='in', which='major',length=10)
plt.xlabel(r'$t$ $(ns)$',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(np.arange(y.min(),y.max(),1,dtype=int),fontsize=14)
plt.ylabel(r'$N_{chains}$',fontsize=14)
i = 0
v = nt['n_ads_chains']
x = np.array( list(v.keys()))/1000
y = np.array( list(v.values()) )
plt.yticks(np.arange(y.min(),y.max()+1,1,dtype=int),fontsize=14)
plt.plot(x,y, ls = 'none',marker='s',fillstyle='none',markersize=3,label=k,color='blue')
#plt.legend(frameon=False,ncol=2,fontsize=12)
plt.savefig('Nadsorbed_chains_nt.png',bbox_inches='tight')
plt.show() 
 

mda.ass.try_beebbeeb()
mda.ass.clear_logs()



    
