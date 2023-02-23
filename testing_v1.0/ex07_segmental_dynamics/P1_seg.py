# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""


import sys

from matplotlib import pyplot as plt
from time import perf_counter
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import md_analysis as mda  
import numpy as np

##### Setup ####
t0 = perf_counter()

conftype = 'zdir'
#topol_vector = ['C','CD','CD','C']
topol_vector = 4
dads=1.025
conformations = ['train','loop','tail','free']
filt = {'conformations':conformations}
save_data_into_pickle=False

trajf = '../trr/PRwhbench.trr'

traj = mda.Analysis_Confined(trajf,['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'],
                    conftype, 
                    topol_file ='../gro/alupb.gro',
                    particle='ALU',polymer='PB')

traj.read_file()

#STEP 1: find segmental vectors and their population at each time
seg_t, fseg = traj.calc_segmental_vectors_t(topol_vector,
        filters = filt,
        dads=dads)
#seg_t is dictionary
#keys --> time floats 
#values --? np.arrays with the segmental vectors at that time
#fseg is dictionary 
#keys --> population strings like train,tail,loop etc ...
#values --> dictionaries similar to seg_t
            #keys --> same as seg_t
            #values --> boolean arrays of the same size as seg_t values. Each vector gets a True if it belongs to the population at time t and false if not
            
#dyn = {k : traj.Dynamics('P1',seg_t,fs) for k,fs in fseg.items()}

#STEP 2:
#Now we wish to compute segmental dynamics of each population
#We will store them in a dictionary. We can save them later in pickle form and plot them easily
#way one
dyn = dict()
dyn['system'] = traj.Dynamics('P1',seg_t)
 # to compute the dynamics of the whole system I give seg_t, without any boolean data
# to distinguish the population we give the boolean data
dyn['train'] = traj.Dynamics('P1',seg_t,fseg['train'])
dyn['loop'] = traj.Dynamics('P1',seg_t,fseg['loop'])
dyn['tail'] = traj.Dynamics('P1',seg_t,fseg['tail'])
dyn['free'] = traj.Dynamics('P1',seg_t,fseg['free'])
#more compact way to to the same as the previoues 8 lines
#dyn = {k : traj.Dynamics('P1',seg_t,fs) for k,fs in fseg.items()}
#dyn['system'] = traj.Dynamics('P1',seg_t) 


#We will also add the dynamics of bulk
bulk = mda.Analysis_Confined('../trr/bulkPRwh.trr',
                             '../itp/topol_UA_PB30.itp',
                        conftype, 
                        topol_file ='../gro/bulk.gro',
                        particle='PB',polymer='PB')
bulk.read_file()

vect,fbulkseg = bulk.calc_segmental_vectors_t(topol_vector)
dyn['bulk'] = bulk.Dynamics('P1',vect)
    
##Notice!!! that we need short time data. On ex11 we will show how to wrap two trajectories with different time intervals



size = 3.5
figsize = (size,size)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=size*1.5)
plt.tick_params(direction='in', which='major',length=size*3)
plt.xscale('log')
plt.xticks(fontsize=2.5*size)
plt.yticks(fontsize=2.5*size)
plt.xlabel(r'$t (ns)$',fontsize=3*size)
plt.ylabel(r'$P_1(t)$',fontsize=3*size)
for i,(k,dy) in enumerate(dyn.items()):    
    y = mda.ass.numpy_values(dy) # make the values of the dictionary array
    x = mda.ass.numpy_keys(dy)/1000
    x = x[1:]
    y = y[1:]
    args = mda.plotter.sample_logarithmically_array(x,num=80) # use this to sabsample logarithmically your data
    plt.plot(x[args],y[args],ls='none',marker = 'o',markeredgewidth=0.2*size,
        markersize=size*1.2,fillstyle='none',label=k)
plt.legend(frameon=False,fontsize=2.5*size)
plt.savefig('segvP1.png',bbox_inches='tight')
plt.show()

#save your data into pickle
if save_data_into_pickle:
    import pickle
    data_file ='my_segmental_dataP1.pickle'
    with open(data_file,'wb') as handle:
        pickle.dump(dyn, handle, protocol=pickle.HIGHEST_PROTOCOL)


mda.ass.print_time(perf_counter()-t0, 'MAIN',1000)
mda.ass.try_beebbeeb()
mda.ass.clear_logs()