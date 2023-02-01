# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""

import sys
from matplotlib import pyplot as plt
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')

import md_analysis as mda  
import numpy as np
##### Setup ####

trajf = '../trr/PRwh_dt1.trr' # contains about 1800 frames
conftype = 'zdir'
binl=0.2 ; dmax =5 ; 

#readings

confined = mda.Analysis_Confined(trajf,['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'],
                        conftype, 
                        topol_file ='../gro/alupb.gro',
                        particle='ALU',polymer='PB')

confined.read_file()

bulk = mda.Analysis_Confined('../trr/bulkPRwh.trr','../itp/topol_UA_PB30.itp',
                             conftype,
                             topol_file='../gro/bulk.gro',
                             particle='PB',polymer='PB' # NOTICE! I use PB on both particle and PB. PB center of mass would be considered insted of some particles, i.e. center of box for bulk
                             )
bulk.read_file()


# calculations
chain_chars = confined.calc_chain_characteristics(0.5,dmax,binl)
bulk_chars = bulk.calc_chain_characteristics(0.5,4.5,binl)
# store in shorter variables
d = chain_chars
b = bulk_chars



# I get the particle thickness on z axis and divide by two to find the distance from the surface and not the substrate center of mass
t2 = confined.calc_particle_size()[2]/2 
d['d'] -= t2


figsize = (3.5,3.5)
dpi = 300
fig,host =plt.subplots(figsize=figsize,dpi=dpi)
par1 = host.twinx()
caxis='green'
host.minorticks_on()
host.tick_params(direction='in', which='minor',length=5)
host.tick_params(direction='in', which='major',length=10)
plt.xlim([0,dmax-t2])
host.set_xlabel(r'$d (nm)$')
#host.set_ylabel(r'$<Rg^2_{xy}>$/$<Rg^2>$')
host.plot(d['d'],d['Rg'],label=r'$<Rg>$',color='k',
          marker='o',fillstyle='none',ls='--',markersize=5)
par1.plot(d['d'],d['Rgxx_plus_yy']/d['Rg2'],label=r'$<Rg^2_{xy}>$/$<Rg^2>$',
         marker='o',fillstyle='none',ls='--',markersize=5,color=caxis)

Rgb = bulk_chars['Rg'].mean()
Rgb_std = bulk_chars['Rg'].std()
ds = d['d'][np.logical_and(d['d']>=0.2, d['d']<1.2)]
n = ds.shape[0]
bm = (b['Rgxx_plus_yy']/b['Rg2']).mean()
bs = (b['Rgxx_plus_yy']/b['Rg2']).std()
par1.set_ylim([0.6,1.02])
par1.errorbar([3.5],[bm],yerr=[bs],ls='none',marker='s',fillstyle='none',
          markersize=6,color=caxis,capsize=5,label=r'($<Rg^2_{xy}>/<Rg^2>)^{bulk}$')
host.errorbar([2.2],[Rgb],yerr=[Rgb_std],ls='none',marker='s',fillstyle='none',
          markersize=6,color='k',capsize=5,label=r'$Rg^{bulk}$')
par1.tick_params(axis='y',colors=caxis)
par1.spines["right"].set_edgecolor(caxis)
host.set_ylabel(r'$nm$',color='k')
host.legend(frameon=False,loc='center right',labelcolor='k')
par1.legend(frameon=False,loc='upper right',fontsize=9,labelcolor=caxis)
plt.savefig('Rgxx_plus_yy_Rg2.png',bbox_inches='tight')
plt.show()


fig =plt.figure(figsize=figsize,dpi=dpi)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=5)
plt.tick_params(direction='in', which='major',length=10)
plt.xlim([0,dmax-t2])
plt.xlabel(r'$d (nm)$')
#plt.ylim([,])
k2n= bulk_chars['k2'].mean()
acyln = bulk_chars['acyl'].mean()
asphn = bulk_chars['asph'].mean()
plt.plot(d['d'],d['k2']/k2n,label=r'anisotropy',
         marker='o',fillstyle='none',ls='--',markersize=5,color='k')
plt.plot(d['d'],d['acyl']/acyln,label=r'acylindricity',
         marker='o',fillstyle='none',ls='--',markersize=5,color='grey')
plt.plot(d['d'],d['asph']/asphn,label=r'asphericity',
         marker='o',fillstyle='none',ls='--',markersize=5,color='orange')
plt.plot(d['d'],[1]*d['d'].shape[0],ls='--',color='k',lw=0.5)
plt.legend(frameon=False)
plt.savefig('shape.png',bbox_inches='tight')
plt.show()
print('Bulk k2: {:4.5f} \nBulk acyl: \
      {:4.5f}\nBulk asph: {:4.5f}'.format(k2n,acyln,asphn))



fig,host =plt.subplots(figsize=figsize,dpi=dpi)
par1 = host.twinx()
caxis='green'
host.minorticks_on()
host.tick_params(direction='in', which='minor',length=5)
host.tick_params(direction='in', which='major',length=10)
plt.xlim([0,dmax-t2])
host.set_xlabel(r'$d (nm)$')
host.set_ylabel(r'$nm$')
host.set_ylim([3,4.6])
host.spines['left'].set_edgecolor('k')
host.plot(d['d'],d['Ree2']**0.5,label=r'$<Ree^2>^{1/2}$',color='k',
          marker='o',fillstyle='none',ls='--',markersize=5)

Reeb = np.sqrt(bulk_chars['Ree2']).mean()
Reeb_std = np.sqrt(bulk_chars['Ree2']).std()
ds = d['d'][np.logical_and(d['d']>= 2.1, d['d']<3)]
n = ds.shape[0]
host.plot([ds.mean(),ds.mean()],[Rgb+Rgb_std,Rgb-Rgb_std],ls='--',lw=0.7,color=caxis)
host.plot(ds,[Reeb]*n+Reeb_std,ls='--',lw=0.7,color='k')
host.plot(ds,[Reeb]*n-Reeb_std,ls='--',lw=0.7,color='k')
host.plot([ds.mean()],[Reeb],ls='none',marker='s',fillstyle='none',
          markersize=6,color='k',label=r'$<Ree^2>^{1/2}_{bulk}$')
host.plot([ds.mean(),ds.mean()],[Reeb+Reeb_std,Reeb-Reeb_std],ls='--',lw=0.7,color='k')
par1.plot(d['d'],d['Rg'],label=r'$<Rg>$',color=caxis,
          marker='o',fillstyle='none',ls='--',markersize=5)

#par1.errorbar(d['d'],d['Rg'],yerr=d['Rg(std)'])
par1.tick_params(axis='y',colors=caxis)
par1.spines["right"].set_edgecolor(caxis)
par1.set_ylabel(r'$Rg(nm)$',color=caxis)
host.legend(frameon=False,loc='upper right')
plt.savefig('Ree.png',bbox_inches='tight')
plt.show()




    