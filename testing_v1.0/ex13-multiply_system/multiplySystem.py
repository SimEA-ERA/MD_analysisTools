# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""

import os
import sys

from matplotlib import pyplot as plt
import os
import numpy as np

sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import md_analysis as mda  



obj = mda.Analysis('confout.gro','topol_UA_PB30.itp')

obj.read_file()
multiplicity=(2,2,0) 
fname = 'mult_x{}y{}z{}.gro'.format(*multiplicity)
obj.multiply_periodic(multiplicity)
obj.write_gro_file(fname=fname)
mda.ass.try_beebbeeb()
mda.ass.clear_logs()