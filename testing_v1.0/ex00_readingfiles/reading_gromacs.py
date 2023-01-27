# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""
# you need to have md_analysis in the current path,
# or your python packages path, e.g. C:\Users\n.patsalidis\Anaconda3\ 
# or insert the path of the file like below
'''
import sys
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
'''

import md_analysis as mda  
# Reading a bulk system
analysis_obj_bulk = mda.Analysis('../trr/bulkPRwh.trr', # trajectory file
                                 '../itp/topol_UA_PB30.itp', # connectivity file
                                 '../gro/bulk.gro' # topology file
                                 )


analysis_obj_bulk.read_file() # reads the trajectory, stores data in a dictionary called timeframes
#Reading a confined system

trajf = '../trr/PRwh_dt1.trr' # trajectory file
conftype = 'zdir' #type of confinmnent
connectivity_info = ['../itp/topol_UA_PB30.itp','../itp/alu_to.itp'] # bond information

analysis_obj_confined = mda.Analysis_Confined(trajf, #trajectory file
                        connectivity_info, # can be either a list of files or just one file
                        conftype, # signifies what functions to use to calculate e.g. the distance, or the volume of each bin
                        topol_file ='../gro/alupb.gro', # if it's gromacs setup we need a gro file of one frame to read atom types, molecule types and exetra 
                        particle='ALU',polymer='PB') # Need to give the particle and polymer name 

analysis_obj_confined.read_file()

#access your data
timeframe0 = analysis_obj_confined.timeframes[0]
#frame 10 coords for some reason
c10 = analysis_obj_confined.get_coords(10)
#frame 10 time for some reason
t10 = analysis_obj_confined.get_time(10)
#or
t10_otherwise = analysis_obj_confined.timeframes[10]['time']


