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
     ('loosely dotted',        (0, (1, 3))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 2))),

     ('loosely dashed',        (0, (5, 3))),
     ('dashed',                (0, (4, 2))),
     ('densely dashed',        (0, (3, 1))),

     ('loosely dashdotted',    (0, (5, 3, 1, 3))),
     ('dashdotted',            (0, (4, 2, 1, 2))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (5, 2, 1, 2, 1, 2))),
     ('loosely dashdotdotted', (0, (4, 3, 1,3, 1, 3))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]*10
colors = ['#b2182b','#d6604d','#f4a582','#fddbc7','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac']
lst =[l[1] for l in linestyle_tuple]
size = 3.5
figsize = (size,size)
path = 'Exersize4_figures'
make_dir(path)
save_figs = True
dpi=300
import md_analysis as mda
import numpy as np
from matplotlib import pyplot as plt
funcs = mda.Analytical_Expressions()
dt = 0.01
maxt=1000
t0s = [1]
Ass = [0.5,1]
Aee = [0.5,1.5]
figsize = (size,size)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=size*1.5)
plt.tick_params(direction='in', which='major',length=size*3)
plt.xscale('log')
plt.xticks(fontsize=2.5*size)
plt.yticks(fontsize=2.5*size)
plt.ylabel(r'$e^{-t/t0}$ ',fontsize=size*3)
plt.xlabel(r'$t$',fontsize=size*3)
t= np.arange(dt,maxt+dt,dt)
for j,As in enumerate(Ass):
    for i,Ae in enumerate(Aee):
        y = funcs.expDecay4(t,As,Ae,1)
        plt.plot(t,y,ls=lst[i*4],lw=3,color=colors[j*5],
         label=r'$A_s$={},$A_e$={}'.format(As,Ae))
plt.legend(frameon=False,fontsize=size*2.5)
if save_figs:plt.savefig(path +'/expdecay4_AsAe.png',bbox_inches='tight')
plt.show()

