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

def check_surf(obj,num,ty='S09'):
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
    'cp confout.gro minimization.gro',
    '#rm tp.tpr',
    '#gmx grompp -c confout.gro  -f gromp.mdp -p {:s}.top -o tp.tpr'.format(sysname) ,
    '#gmx mdrun -s tp.tpr',] 
    with open(fname,'w') as f:
        for line in lines:
            f.write('{:s}\n'.format(line))
        f.close()
    return
def write_n_run(mol,num):
    if False:
        sil.element_based_matching([('O02R','SiR'),('Si3', 'O02R', 'SiR'),('O02R', 'SiR', 'Ob'),('O02R', 'SiR', 'Onb')])
    else:
        if mol in ['MPTES','TESPD']:
            sil.match_types([ ('O02R','SiR'), ('Si3', 'O02R', 'SiR'), ('O02R', 'SiR', 'Ob'),('O02R', 'SiR', 'Onb') ],
                             [ ('Ob','Si'),  ('Si', 'Ob', 'Si') ,      ('Ob', 'Si', 'Ob'),    ('Ob', 'Si', 'Ob') ] )
            #sil.ff.bondtypes[('O02R','SiR')] = sil.ff.bondtypes[('Ob','Si')]
            #sil.ff.angletypes[('Si3', 'O02R', 'SiR')] = sil.ff.angletypes[('Si', 'Ob', 'Si')]
            #sil.ff.angletypes[('O02R', 'SiR', 'Ob')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]
            #sil.ff.angletypes[('O02R', 'SiR', 'Onb')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]  
        else:
            sil.ff.bondtypes[('O0KR','SiR')] = sil.ff.bondtypes[('Ob','Si')]
            sil.ff.angletypes[('SiD', 'O0KR', 'SiR')] = sil.ff.angletypes[('Si', 'Ob', 'Si')]
            sil.ff.angletypes[('O0KR', 'SiR', 'Ob')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]
            sil.ff.angletypes[('O0KR', 'SiR', 'Onb')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]
    sil.filter_ff()
    sil.clean_dihedrals_from_topol_based_on_ff()
    
    sysname = 'S{:s}{:d}'.format(mol[0],num)
    sil.exclusions_map[sysname] = 2
    sil.exclusions = sil.pairs # this is a trick to not count twice the 1-4 interactions
    sil.mol_names[:] = sysname
    sil.mol_ids[:] = 1
    
    
    tw = perf_counter()
    sil.write_topfile('{:s}/{:s}'.format(savepath,sysname),opls_convection=True)
    sil.write_gro_file('{:s}/{:s}.gro'.format(savepath,sysname))
    
    write_bash('{:s}/{:s}.sh'.format(savepath,sysname),sysname)
    print('writing time = {:.3e} sec'.format(perf_counter()-tw))
    com1 = 'cp setup_eq_functionilization/*mdp {:s}'.format(savepath)
    com2 = 'cd {:s}'.format(savepath)
    com3 = 'bash {:s}.sh'.format(sysname)
    
    
    tr = perf_counter()
    command = ' ; '.join([com1, com2, com3])
    print(command)
    os.system(command)
    os.system('cp visu1.vmd {:s}/'.format(savepath) )
    print('Running time = {:.3e} sec'.format(perf_counter()-tr))
    return 
#Settings
########################################################################################
####################################    
#cases = [1,2,3]
silc = 'Kempfer_Slab/Silica_slab_Kempfer.gro'
silt = 'Kempfer_Slab/Silica_slab_Kempfer.top'
bonds = {'MPTES':('C01','O02'),'TESPD':('C01','O02'),'NXT':('C0M','O0K')}

cite_method = 'uniform'
grid = (3,3)
nwrite=2

cite_method_kwargs = {'random':dict(),
                      'height':dict(),
                      'height_neibs':dict(),
                      'separation_distance':dict(separation_type='SiR',separation_distance=0.8),
                      'uniform':dict(separation_type='SiR',separation_distance=0.8,grid=grid)
                      } 
c = ('Onb','Si')
morse_bond={'NXT':(100,0.16,2),
            'TESPD':(100,0.16,2),
            'MPTES':(100,0.16,2)}
morse_overlaps = {'NXT':(0.2,5),
                'TESPD':(0.2,5),
                'MPTES':(0.2,5),
                 }
max_target_density = 1.77
for mol in ['MPTES','NXT','TESPD']:

    
    sil = mda.Analysis(silc,silt)
    sil.read_file(sil.topol_file)
    
    
    surf_area = 2*np.prod(sil.get_box(0)[:2])
    num = int(round(surf_area*max_target_density,0))
    
    td = num/surf_area
    savepath = 'SiO2-{:s}/{:2.2f}/'.format(mol,td)

    mda.ass.make_dir(savepath)
    
    
    if num%2==1: num+=1
    
    coupling_agent = mda.Analysis('ligpargen/{:s}.gro'.format(mol),
                                  'ligpargen/{:s}.itp'.format(mol))
    #raise
    coupling_agent.read_file(coupling_agent.topol_file)
    
    rcut = 4*Rg(coupling_agent.get_coords(0))
    
    print(rcut)    
    t0 = perf_counter()
    for i in range(num):
        print('Adding molecule {:s} {:d}/{:d}'.format(mol,i+1,num))
        coupling_agent = mda.Analysis('ligpargen/{:s}.gro'.format(mol),
                                  'ligpargen/{:s}.itp'.format(mol))
        coupling_agent.read_file(coupling_agent.topol_file)
        a = mda.React_two_systems(sil,coupling_agent,
                                  c,bonds[mol],
                                  react1=1,react2=1,
                                  seed1=None,seed2=None,
                                  rcut=rcut,
                                  morse_bond = morse_bond[mol],
                                  morse_overlaps = morse_overlaps[mol],
                                  bound_types=['H','Onb'],
                                  cite_method = cite_method,
                                  cite_method_kwargs=cite_method_kwargs[cite_method])
        tf = perf_counter() - t0
        print(' added {:d} molecules time --> {:.3e}  sec  time/mol = {:.3e} sec/mol'.format(i+1,tf,tf/(i+1)))
        if (i+1) % nwrite ==0 or (i+1)==num:
            sysname = 'S{:s}{:d}'.format(mol[0],i+1)
            write_n_run(mol,i+1)
            gro_file = '{:s}/confout.gro'.format(savepath)
            del sil.timeframes[0]
            sil.read_file(gro_file)
            subsys = '{:s}/{:s}'.format(savepath,sysname)
            mda.ass.make_dir(subsys)
            os.system('mv {:s}/*.* {:s}'.format(savepath,subsys))
    
    #write_n_run(mol,num)
        
    
    
    

#os.system('cd {:s}/{:s} ; bash eqall.sh'.format(mol,dens))