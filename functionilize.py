# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:50:42 2023

@author: n.patsalidis
"""

import sys

sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import md_analysis as mda  
import numpy as np
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
silc = 'Kempfer_Slab/Silica_slab_Kempfer.gro'
silt = 'Kempfer_Slab/Silica_slab_Kempfer.top'
bonds = {'MPTES':('C01','O02'),'TESPD':('C01','O02'),'NXT':('C0M','O0K')}
mol = 'MPTES'
target_density = 1.54
sil = mda.Analysis(silc,silt)
sil.read_file(sil.topol_file)
surf_area = 2*np.prod(sil.get_box(0)[:2])

num = int(round(surf_area/target_density,0))

#num = 10
c = ('Onb','Si')

for i in range(num):
    print('Adding molelecule {:s} {:d}'.format(mol,i))
    coupling_agent = mda.Analysis('ligpargen/{:s}.gro'.format(mol),
                              'ligpargen/{:s}.itp'.format(mol))
    coupling_agent.read_file(coupling_agent.topol_file)
    a = mda.React_two_systems(sil,coupling_agent,
                              c,bonds[mol],
                              react1=1,react2=1,
                              seed1=None,seed2=None,rcut=2)
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
sil.write_topfile(sysname)
sil.write_gro_file('{:s}.gro'.format(sysname))