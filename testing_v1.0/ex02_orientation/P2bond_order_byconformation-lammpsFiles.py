# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""

##### Setup ####
code_path = ''
trajf = '../lammps_files/trajectory.lammpstrj' # trajectory
confinement_type = 'spherical_particle'
topol_file = '../lammps_files/centered.ltop' # copy your topology file to have .ltop extention 
binl=0.025 # length of binning
dmax = 4.5 # maximum distance
dads = 2.5 # adsorbed layer distance 
topol_vector = 4 # gets all 1-4 vectors
particle_name = '34' # particle name
conf = 'tail' # 'bridge','loop','train','tail','free' # conformation to calculate orientation
polymer_first_id = 1 # polymer first id (included)
polymer_last_id = 33 # polymer last id (included)
#################


#MAIN CODE
import sys
from matplotlib import pyplot as plt

sys.path.insert(0, code_path)

import md_analysis as mda  
import numpy as np

#reading the system
polymer_ids = np.arange(polymer_first_id,polymer_last_id+1,1,dtype=int)

obj = mda.Analysis_Confined(trajf, #lammps trajectory file
                             topol_file, # lammps topology file
                             confinement_type, 
                             polymer_method ='molids', # method which are the polymer chains
                             particle=particle_name, # name of particle
                             polymer=polymer_ids #polymer ids
                             )
obj.read_file()


#caclulating P2
P2 = obj.calc_P2(binl,dmax,topol_vector,dads=dads,option=conf)

#write_file
with open('P2_{}.txt'.format(conf),'w') as f:
    for d,p2 in zip(P2['d'],P2['P2']):
        f.write('{:8.8f}  {:8.8f} \n'.format(d,p2))
    f.closed

#plotting
figsize=(3.3,3.3)
dpi=300
fig =plt.figure(figsize=figsize,dpi=dpi)
plt.title(conf)
plt.minorticks_on()
plt.tick_params(direction='in', which='minor',length=5)
plt.tick_params(direction='in', which='major',length=10)
plt.xlim([0,dmax])

plt.plot(P2['d'],P2['P2'],
        ls='none',marker='o',fillstyle='none',
        markersize=4,color='k',lw=1.5)

plt.xlabel(r'd $(nm)$')
plt.ylabel(r'$P_2(\theta)$')
plt.legend(frameon=False)
plt.savefig('P2_{}.png'.format(conf),bbox_inches='tight')
#plt.show()


    