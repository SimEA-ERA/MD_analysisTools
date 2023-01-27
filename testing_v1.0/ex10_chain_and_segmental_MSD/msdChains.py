# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 08:49:55 2022

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
filt= {'adsorption':None}  


trajf='../trr/PRnj_dt1.trr' # no-jump coordinates!!
results = mda.Analysis_Confined(trajf,['../itp/topol_UA_PB30.itp',
                                       '../itp/alu_to.itp'],
                conftype, 
                topol_file ='../gro/alupb.gro',
                particle='ALU',polymer='PB')


results.read_file()
#center of mass per time, filter per time
chcm_t, filt_t = results.calc_chainCM_t(filters = filt,dads=dads)

MSD = dict()
for di in ['','xy-','z-']: # we decompose each direction
    
    cmt = {k:v.copy() for k,v in chcm_t.items()} # we first make a copy
    
    if di =='': pass # if no direction we don't change to zero the center of mass data
    elif di =='xy-':
        for t in cmt:
            cmt[t][:,2] = 0
    elif di == 'z-':
        for t in cmt:
            cmt[t][:,0:2] = 0
    
    MSD[di+'ads']= results.Dynamics('MSD',cmt, 
                                    mda.ass.stay_True(filt_t['ads']),filt_option='strict')
    MSD[di+'free']= results.Dynamics('MSD',cmt, 
                                    mda.ass.stay_True(filt_t['free']),filt_option='strict')

    MSD[di+'system'] = results.Dynamics('MSD',cmt) #here we pass with no filter
    

#Plots #########
ls = mda.ass.linestyles.lst_map
lsmap = {'xy':ls['densely dashed'],'z':ls['loosely dotted']}
c = mda.ass.colors.qualitative.colors6
cmap = {'ads':c[0],'free':c[3],'bulk':c[5],'system':c[4]}
MSDn = {k : v for k,v in MSD.items() if  'xy' in k and 'degree' not in k}
MSDn.update({k : v for k,v in MSD.items() if  'z' in k and 'degree' not in k} )

cmap = {k:cmap[k.split('-')[-1]] for k in MSDn}
pmap = {k:'o' if 'xy'  in k else 's' for k in MSDn}
fname = 'MSDxy_vs_z_chains.png'
lstmap = lsmap = {k: ls['densely dashed'] if 'xy' in k else ls['loosely dotted'] for k in MSDn}
msdnorm = {k:{t:x/2 if 'xy' in k else x for t,x in v.items() } for k,v in MSDn.items()} 
labmap = {k:'1/2'+k if 'xy' in k else k for k in msdnorm}

mda.plotter.plotDynamics(msdnorm,fname = fname,style='lines',lstmap=lstmap,
                         ylabel = r'$MSD(t)$ / $nm^2$ ',labmap=labmap,
                         pmap=pmap,ncolleg=2,ylim=(-5,3),xlim=(-4,4),
                         cmap = cmap,yscale='log',num=50)

mda.ass.print_time(perf_counter()-t0, 'MAIN',1000)
#mda.ass.try_beebbeeb()
mda.ass.clear_logs()
