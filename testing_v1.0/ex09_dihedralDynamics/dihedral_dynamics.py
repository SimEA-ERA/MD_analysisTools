# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:12:46 2022

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


##### Setup ####

colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']*10
colors = ['#d73027','#f46d43','#fdae61','#fee090','#abd9e9','#74add1','#4575b4']*10

conftype = 'zdir'

dads=1.025
dihedral = 'alpha'
phib = ('CD','C','C','CD')
phia = ('C','C','CD','CD')

if dihedral == 'beta':
    phi = phib
elif dihedral == 'alpha':
    phi = phia


##################
trajf = '../trr/PRwh_dt100fs.trr'
results = mda.Analysis_Confined(trajf,['../itp/topol_UA_PB30.itp',
                                       '../itp/alu_to.itp'],
                conftype, 
                topol_file ='../gro/alupb.gro',
                memory_demanding=True, # will delete each frame after calculation
                particle='ALU',polymer='PB')


filt = {'conformations':['train','loop','tail','free']}

phi_t, fphi_t = results.calc_dihedrals_t(phi,
            filters=filt,dads=dads)
    
# use the dictionary comprehension
phidict = {k: results.TACF('sin',phi_t,fs) for k,fs in fphi_t.items()}   
  

# see documentation for filt option 
# TACF is a time autocorrelation function for scalar variables. Passing sin takes the sin of the scalar variable
phidict[r'train (at $\tau_0,t$)'] =results.TACF('sin',phi_t,fphi_t['train'],filt_option='strict',) 
phidict['train (const)'] =results.TACF('sin',phi_t,fphi_t['train'],filt_option='const') 
phidict['train (changed)'] =results.TACF('sin',phi_t,fphi_t['train'],filt_option='change')
    
 
# set variables for plotter 
fname = 'phi_{:s}.png'.format(dihedral)

if dihedral =='alpha':
    cutf ={'train':1100,
           'loop':10,
           'tail':5,
           'free':1,#
           'train (at $\\tau_0,t$)':500,
          'train (const)':1000,
          'train (changed)':1000}
elif dihedral =='beta':
    cutf ={'train':1100,
           'loop':10,
           'tail':5,
           'free':1,#
           'train (at $\\tau_0,t$)':900,
          'train (const)':500,
          'train (changed)':1000,
          }
ylab ='{'+'\phi_\{}(t)'.format(dihedral)+'}'

ylabel = r'$TACF({})$ '.format(ylab)

phi = {k:v for k,v in phidict.items()}

mda.plotter.plotDynamics(phi,fname=fname,ylabel=ylabel,
                         xlim=(-4,3),cutf=cutf)

mda.ass.try_beebbeeb()
mda.ass.print_time(perf_counter()-t0, 'MAIN',1000)



    