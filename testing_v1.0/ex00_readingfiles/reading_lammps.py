"""
Created on Sat Jun 25 22:37:42 2022

@author: n.patsalidis
"""

import md_analysis as mda  
import numpy as np

     # we give a polymer_method because lammps trajectory topologies
     # do not contain info about the name of the polymer
     # we give polymer_method = 'molids' and we give all the molecule ids of our polymer
     #For the particle we give the name, which by default is equal to the particle mol id but it is a string
traj = mda.Analysis_Confined('../lammps_files/trajectory.lammpstrj', #lammps trajectory file
                             '../lammps_files/centered.ltop', # lammps topology file
                              'spherical_particle', #confinemt type
                              polymer_method ='molids',
                              particle='34',
                              polymer=np.arange(0,34,1,dtype=int)
                              )
traj.read_file()

coords5 = traj.get_coords(5)
