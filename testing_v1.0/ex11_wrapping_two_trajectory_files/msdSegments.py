# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 08:49:55 2022

@author: n.patsalidis
"""

import sys
from matplotlib import pyplot as plt
import matplotlib


from time import perf_counter
sys.path.insert(0, '\\Users\\n.patsalidis\\Desktop\\PHD\\REPOSITORIES\\MDanalysis')
t0 = perf_counter()
import md_analysis as mda  
import numpy as np
conftype = 'zdir'
dads=1.025
t0 = perf_counter()



    
conformations = ['train','loop','tail','free']
filter_wr_segcm = {'conformations':conformations}  

def msdSegments(trajf):
    MSD = dict()

    results = mda.Analysis_Confined(trajf,['../itp/topol_UA_PB30.itp',
                                       '../itp/alu_to.itp'],
                conftype, 
                topol_file ='../gro/alupb.gro',
                particle='ALU',polymer='PB')
    results.read_file()
    

    segcm_t, filt_t = results.calc_segCM_t(['C','CD','CD','C'],('C','C')
                                          ,filters = filter_wr_segcm,dads=dads)
    for di in directions:
        st = {k:v.copy() for k,v in segcm_t.items()}
        if di =='tot':
            lab = ''
        elif di =='xy':
            lab = 'xy-'
            for t in st:
                st[t][:,2] = 0
        elif di =='z':
            lab = 'z-'
            for t in segcm_t:
                st[t][:,0:2] = 0
   
        MSD.update({ lab+k :results.Dynamics('MSD',
                    st,mda.ass.stay_True(fa),filt_option='strict'
                                            )
                    for k,fa in filt_t.items() 
                    }
                   )

    return MSD

    
directions = ['tot','xy','z']
#this function wraps the msdSegments function and merges properly it's return Values
MSD =  mda.ass.wrap_multiple_trajectory(
['../trr/PRnj_dt100fs.trr','../trr/PRnj_dt1.trr'],
    msdSegments)
        
    
        


################
#Plots #########

ls = mda.ass.linestyles.lst_map

c = mda.ass.colors.qualitative.colors6
cmap = {'train':c[0],'loop':c[1],'tail':c[2],'free':c[3],'bulk':c[5]}
MSDn = {k : v for k,v in MSD.items() if  'xy' in k }
MSDn.update({k : v for k,v in MSD.items() if  'z' in k })


ls = mda.ass.linestyles.lst_map
lsmap = {'xy':ls['densely dashed'],'z':ls['loosely dotted']}
MSDn = {k : v for k,v in MSD.items() if  'xy' in k and 'degree' not in k}
MSDn.update({k : v for k,v in MSD.items() if  'z' in k and 'degree' not in k} )

cmap = {k:cmap[k.split('-')[-1]] for k in MSDn}
pmap = {k:'o' if 'xy'  in k else 's' for k in MSDn}
fname = 'MSDxy_vs_z_seg.png'
lstmap = lsmap = {k: ls['densely dashed'] if 'xy' in k else ls['loosely dotted'] for k in MSDn}
msdnorm = {k:{t:x/2 if 'xy' in k else x for t,x in v.items() } for k,v in MSDn.items()} 
labmap = {k:'1/2'+k if 'xy' in k else k for k in msdnorm}

#use the keyword midtime to sample correctly logarithmically both merged trajectories
mda.plotter.plotDynamics(msdnorm,fname = fname,style='lines',lstmap=lstmap,
                         ylabel = r'$MSD(t)$ / $nm^2$ ',labmap=labmap,
                         midtime=1,pmap=pmap,ncolleg=2,ylim=(-5,3),xlim=(-4,4),
                         cmap = cmap,yscale='log',num=50)

     
mda.ass.print_time(perf_counter()-t0, 'MAIN',1000)
mda.ass.try_beebbeeb()
mda.ass.clear_logs()
