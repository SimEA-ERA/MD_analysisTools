# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:56:07 2022

@author: n.patsalidis
"""
import os 

def make_dir(path):
    try:
        if not os.path.exists(path):
            x = os.system('mkdir ' + path)
            if x != 0:
                raise ValueError
        else:
            return
    except:
        path = '\\'.join(path.split('/'))
        make_dir(path)
    else:
        print('Created DIR: '+path)
    return
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 1))),
     ('dotted',                (0, (1,1,3, 1))),
     ('densely dotted',        (0, (2,2,3, 3))),

     ('loosely dashed',        (0, (5, 3))),
     ('dashed',                (0, (4, 2))),
     ('densely dashed',        (0, (3, 1))),

     ('loosely dashdotted',    (0, (5, 3, 1, 3))),
     ('dashdotted',            (0, (4, 2, 1, 2))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (5, 2, 1, 2, 1, 2))),
     ('loosely dashdotdotted', (0, (4, 3, 1,3, 1, 3))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]*10
colors = ['#d7191c','#fdae61','#ffffbf','#abdda4','#2b83ba']
lst =[l[1] for l in linestyle_tuple]
size = 3.5
figsize = (size,size)
path = 'Exersize7_figures'
make_dir(path)
save_figs = True
dpi=300
import md_analysis as mda
import numpy as np
from matplotlib import pyplot as plt
funcs = mda.Analytical_Expressions()
dt = 0.01
maxt=1000
t0s = [[1],[1,10,100]]
beta = [[1],[1,5],[0.1,1],[0.1,0.2,0.3,0.5]]
figsize = (size,size)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=size*1.5)
plt.tick_params(direction='in', which='major',length=size*3)
plt.xscale('log')
plt.xticks(fontsize=2.5*size)
plt.yticks(fontsize=2.5*size)
plt.ylabel(r'$KWW$ ',fontsize=size*3)
plt.xlabel(r'$t$',fontsize=size*3)
t= np.arange(dt,maxt+dt,dt)
for i,t0 in enumerate(t0s):
    for j,b in enumerate(beta):
        y = funcs.KWW_sum(t, t0,b)
        plt.plot(t,y,ls=lst[j],lw=2,color=colors[i],
         label=r'$t_0$={},$\beta$={}'.format(t0,b))
plt.legend(frameon=False,ncol=1,bbox_to_anchor=(1,1),fontsize=size*2.5)
if save_figs:plt.savefig(path +'/kWW_sum_t0.png',bbox_inches='tight')
plt.show()
