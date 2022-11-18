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
path = 'Exersize5_figures'
make_dir(path)
save_figs = True
dpi=300
import md_analysis as mda
import numpy as np
from matplotlib import pyplot as plt
funcs = mda.Analytical_Expressions()
dt = 0.01
maxt=1000
twws = [0.1,1,2,5]

figsize = (size,size)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=size*1.5)
plt.tick_params(direction='in', which='major',length=size*3)
plt.xscale('log')
plt.xticks(fontsize=2.5*size)
plt.yticks(fontsize=2.5*size)
plt.ylabel(r'$KWW(A=1,tc=0,beta=1)$ ',fontsize=size*3)
plt.xlabel(r'$t$',fontsize=size*3)
t= np.arange(dt,maxt+dt,dt)
for i,tww in enumerate(twws):
    y = funcs.KWW(t,1,0,1,tww)
    plt.plot(t,y,ls=lst[i],lw=2,color=colors[i],
         label=r'$tww$={}'.format(tww))
plt.legend(frameon=False,ncol=1,fontsize=size*2.5)
if save_figs:plt.savefig(path +'/KWW_vs_tww.png',bbox_inches='tight')
plt.show()


figsize = (size,size)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=size*1.5)
plt.tick_params(direction='in', which='major',length=size*3)
plt.xscale('log')
plt.xticks(fontsize=2.5*size)
plt.yticks(fontsize=2.5*size)
plt.ylabel(r'$KWW(tc=0,beta=1)$ ',fontsize=size*3)
plt.xlabel(r'$t$',fontsize=size*3)
t= np.arange(dt,maxt+dt,dt)
for j,A in enumerate([0.5,1,2]):
    for i,tww in enumerate([0.1,1,5]):
        y = funcs.KWW(t,A,0,1,tww)
        plt.plot(t,y,ls=lst[j],lw=2,color=colors[i],
         label=r'$tww$={},A={}'.format(tww,A))
plt.legend(frameon=False,ncol=1,fontsize=size*2.5)
if save_figs:plt.savefig(path +'/KWW_vs_A_tww.png',bbox_inches='tight')
plt.show()


figsize = (size,size)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=size*1.5)
plt.tick_params(direction='in', which='major',length=size*3)
#plt.xscale('log')
plt.xticks(fontsize=2.5*size)
plt.yticks(fontsize=2.5*size)
plt.ylabel(r'$KWW(tc=0,beta=1)$ ',fontsize=size*3)
plt.xlabel(r'$t$',fontsize=size*3)
t= np.arange(dt,maxt+dt,dt)
plt.xlim([0,1])
for j,A in enumerate([0.5,1,2]):
    for i,tww in enumerate([0.1,1]):
        y = funcs.KWW(t,A,0,1,tww)
        plt.plot(t,y,ls=lst[j],lw=2,color=colors[i],
         label=r'$tww$={},A={}'.format(tww,A))
plt.legend(frameon=False,ncol=1,fontsize=size*2.5)
if save_figs:plt.savefig(path +'/KWW_vs_nolog_A_tww.png',bbox_inches='tight')
plt.show()

dt = 1
figsize = (size,size)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=size*1.5)
plt.tick_params(direction='in', which='major',length=size*3)
plt.xscale('log')
plt.xticks(fontsize=2.5*size)
plt.yticks(fontsize=2.5*size)
plt.ylabel(r'$KWW(A=1,beta=1)$ ',fontsize=size*3)
plt.xlabel(r'$t$',fontsize=size*3)
t= np.arange(dt,maxt+dt,dt)

for j,tc in enumerate([0.5,1,1.5]):
    for i,tww in enumerate([5,10]):
        y = funcs.KWW(t,1,tc,1,tww)
        plt.plot(t,y,ls=lst[j],lw=2,color=colors[i],
         label=r'$tww$={},tc={}'.format(tww,tc))
plt.legend(frameon=False,ncol=1,fontsize=size*2.5)
if save_figs:plt.savefig(path +'/KWW_vs_tc_tww.png',bbox_inches='tight')
plt.show()


dt = 0.01
maxt = 100
figsize = (size,size)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=size*1.5)
plt.tick_params(direction='in', which='major',length=size*3)
plt.xscale('log')
plt.xticks(fontsize=2.5*size)
plt.yticks(fontsize=2.5*size)
plt.ylabel(r'$KWW(A=1,tc=0)$ ',fontsize=size*3)
plt.xlabel(r'$t$',fontsize=size*3)
t= np.arange(dt,maxt+dt,dt)
from scipy.integrate import simpson
for j,b in enumerate([0.5,1,1.5]):
    for i,tww in enumerate([5]):
        y = funcs.KWW(t,1,0,b,tww)
        print('beta = {:}, tchar = {:4.3f}'.format(b,simpson(y,t)))
        plt.plot(t,y,ls=lst[j],lw=2,color=colors[j],
         label=r'$tww$={},b={}'.format(tww,b))
plt.legend(frameon=False,ncol=1,fontsize=size*2.5)
if save_figs:plt.savefig(path +'/KWW_vs_beta_tww.png',bbox_inches='tight')
plt.show()


dt = 0.01
maxt = 100
figsize = (size,size)
dpi = 300
fig=plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=size*1.5)
plt.tick_params(direction='in', which='major',length=size*3)
plt.xscale('log')
plt.xticks(fontsize=2.5*size)
plt.yticks(fontsize=2.5*size)
plt.ylabel(r'$KWW(A=1,tc=0)$ ',fontsize=size*3)
plt.xlabel(r'$t$',fontsize=size*3)
t= np.arange(dt,maxt+dt,dt)
from scipy.integrate import simpson
for j,b in enumerate([0.01,0.1,0.5,1,1.5,2,4,5]):
    for i,tww in enumerate([5]):
        y = funcs.KWW(t,1,0,b,tww)
        print('beta = {:}, tchar = {:4.3f}'.format(b,simpson(y,t)))
        plt.plot(t,y,ls=lst[j],lw=2,color=colors[j],
         label=r'$tww$={},b={}'.format(tww,b))
plt.legend(frameon=False,ncol=1,bbox_to_anchor=(1,1),fontsize=size*2.5)
if save_figs:plt.savefig(path +'/KWW_vs_beta_tww_bigbetarange.png',bbox_inches='tight')
plt.show()


