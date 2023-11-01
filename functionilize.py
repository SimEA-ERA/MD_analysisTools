# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:50:42 2023

@author: n.patsalidis
"""

import sys

sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import md_analysis as mda  
import numpy as np
import os
from time import perf_counter
'''
mass_map = {'Si':28.086,
            'H':1.008,
            'S':32.066,
            'C':12.011,
            'O':15.999400,}

charge_map = {k:0.0 for k in mass_map}

bond_dists = {('Si','O'):(1.5,1.7),
              ('Si','C'):(1.75,2.0),
              ('S','H'):(1.3,1.5),
              ('C','H'):(1.0,1.2),
              ('S','C'):(1.7,1.9),
              ('S','O'):(1,1.5),
              ('C','O'):(1.24,1.3),
              ('C','C'):(1.45,1.6)
              }
bond_dists = {k:(v[0]/10,v[1]/10) for k,v in bond_dists.items()}
'''

def check_surf(obj,num,ty='S07'):
    c = obj.get_coords(0)
    f= obj.at_types == ty
    z = c[:,2]
    zf = z[f]
    
    zr = zf - z.sum()/z.shape[0]
    
    up = np.count_nonzero(zr>0)
    do = np.count_nonzero(zr<0)
    s = '{:d}  {:d}  {:4.3f} {:4.3f}\n'.format(up,do,up/num,do/num)
    return s
def Rg(coords):
    rm = np.mean(coords,axis=0)
    r = coords -rm
    r3 = 0
    for i in range(r.shape[0]):
        r3 += np.dot(r[i],r[i])
    rg = np.sqrt(r3/r.shape[0])
    return rg
def write_bash(fname,sysname):
    lines=['#!/bin/bash',
    'rm tp.tpr',
    'gmx grompp -c {0:s}.gro  -f steep.mdp -p {0:s}.top -o tp.tpr'.format(sysname) ,
    'gmx mdrun -s tp.tpr' ,
    'rm tp.tpr',
    'gmx grompp -c confout.gro  -f gromp.mdp -p {:s}.top -o tp.tpr'.format(sysname) ,
    'gmx mdrun -s tp.tpr',] 
    with open(fname,'w') as f:
        for line in lines:
            f.write('{:s}\n'.format(line))
        f.close()
    return
#Settings
########################################################################################
####################################    
ncases = 10
start = 2

dens ='dens2'
silc = 'Kempfer_Slab/Silica_slab_Kempfer.gro'
silt = 'Kempfer_Slab/Silica_slab_Kempfer.top'
bonds = {'MPTES':('C01','O02'),'TESPD':('C01','O02'),'NXT':('C0M','O0K')}
td = {'MPTES':1.54,'NXT':1.54,'TESPD':0.77}
mol = 'MPTES'
c = ('Onb','Si')
###################################
###################################
########################################################################################
try:
    file_info.close()
except:
    pass
mda.ass.make_dir('{:s}/{:s}'.format(mol,dens))
file_info = open('{:s}/{:s}/surfaceinfo.txt'.format(mol,dens), 'w')
file_info.write('up      down      up_perc     down_perc\n')
file_info.flush()

for case in range(start,ncases+1):
    savepath = '{:s}/{:s}/case{:d}'.format(mol,dens,case)
    mda.ass.make_dir(savepath)

    
    target_density =td[mol]
    
    sil = mda.Analysis(silc,silt)
    sil.read_file(sil.topol_file)
    
    surf_area = 2*np.prod(sil.get_box(0)[:2])
    
    num = int(round(surf_area*target_density,0))
    
    coupling_agent = mda.Analysis('ligpargen/{:s}.gro'.format(mol),
                                  'ligpargen/{:s}.itp'.format(mol))
    coupling_agent.read_file(coupling_agent.topol_file)
    
    rcut = 4*Rg(coupling_agent.get_coords(0))
    
    print(rcut)    
    t0 = perf_counter()
    for i in range(num):
        print('Adding molelecule {:s} {:d}/{:d}'.format(mol,i+1,num))
        coupling_agent = mda.Analysis('ligpargen/{:s}.gro'.format(mol),
                                  'ligpargen/{:s}.itp'.format(mol))
        coupling_agent.read_file(coupling_agent.topol_file)
        a = mda.React_two_systems(sil,coupling_agent,
                                  c,bonds[mol],
                                  react1=1,react2=1,
                                  seed1=None,seed2=None,rcut=rcut)
        tf = perf_counter() - t0
        print(' added {:d} molecules time --> {:.3e}  sec  time/mol = {:.3e} sec/mol'.format(i+1,tf,tf/(i+1)))
    if mol in ['MPTES','TESPD']:
        sil.ff.bondtypes[('O02R','SiR')] = sil.ff.bondtypes[('Ob','Si')]
        sil.ff.angletypes[('Si3', 'O02R', 'SiR')] = sil.ff.angletypes[('Si', 'Ob', 'Si')]
        sil.ff.angletypes[('O02R', 'SiR', 'Ob')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]
        sil.ff.angletypes[('O02R', 'SiR', 'Onb')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]  
    else:
        sil.ff.bondtypes[('O0KR','SiR')] = sil.ff.bondtypes[('Ob','Si')]
        sil.ff.angletypes[('SiD', 'O0KR', 'SiR')] = sil.ff.angletypes[('Si', 'Ob', 'Si')]
        sil.ff.angletypes[('O0KR', 'SiR', 'Ob')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]
        sil.ff.angletypes[('O0KR', 'SiR', 'Onb')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]
    sil.clean_dihedrals_on_ff()
    sysname = 'S{:s}{:d}'.format(mol[0],num)
    sil.mol_names[:] = sysname
    sil.mol_ids[:] = 1
    
    s = check_surf(sil,num)
    print(s)
    file_info.write(s)
    file_info.flush()
    sil.write_topfile('{:s}/{:s}'.format(savepath,sysname))
    sil.write_gro_file('{:s}/{:s}.gro'.format(savepath,sysname))
    write_bash('{:s}/{:s}.sh'.format(savepath,sysname),sysname)
    com1 = 'cp setup/*mdp {:s}'.format(savepath)
    com2 = 'cd {:s}'.format(savepath)
    com3 = 'bash {:s}.sh'.format(sysname)
    command = ' ; '.join([com1, com2, com3])
    print(command)
    os.system(command)
    
    
    
    
file_info.close()

#os.system('cd {:s}/{:s} ; bash eqall.sh'.format(mol,dens))