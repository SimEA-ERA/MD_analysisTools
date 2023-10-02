# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:55:40 2023

@author: n.patsalidis
"""
import argparse
import sys
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import md_analysis as mda 
def main():
    help_trajf = "trajectory file(s)\n set denotes average (must be of the same frame length)\n list or tuple denote wrapping the trajectories\n dictionary denotes key : value pairs where the value must be a trajectory file or set or tuple or list of trajectory files"
    help_property = ""
    help_topol = ""
    help_memory_demanding= ""
    help_connectivity=""
    help_keep_frames = ""
    help_conftype = ""
    help_polymer_method = ""
    help_particle_method = ""
    help_polymer = ""
    help_particle = ""
    help_adsorption_interval = ""
    help_train_costum_method = ""
    help_property_args = ""
    help_property_kwargs = "" 
    adddef = " [  default: %(default)s ]"
    
    argparser = argparse.ArgumentParser(description="analyze your system(s)")
    argparser.add_argument('-p',"--property",metavar=None,
            type=str, required=True, help=help_property)
    argparser.add_argument('-f',"--trajf",metavar=None,
            type=str, required=True, help=help_trajf)
    argparser.add_argument('-t',"--topol",metavar=None,
            type=str, required=True, help=help_topol)
    argparser.add_argument('-c',"--connectivity",metavar=None,
            type=str, required=True, help=help_connectivity)
    argparser.add_argument('-mem',"--memory_demanding",metavar=None,
            type=bool, required=False, help=help_memory_demanding)
    argparser.add_argument('-kf',"--keep_frames",metavar=None,
            type=int, required=False,nargs=2, help=help_keep_frames)

    argparser.add_argument('-ct',"--conftype",metavar=None,
            type=str, required=False, help=help_conftype)
    argparser.add_argument('-a',"--adsorption_interval",metavar=None,
            type=str, required=False, help=help_adsorption_interval)
    argparser.add_argument('-pam',"--particle_method",metavar=None,
            type=str, required=False, help=help_particle_method)
    argparser.add_argument('-pom',"--polymer_method",metavar=None,
            type=str, required=False, help=help_polymer_method)
    argparser.add_argument('-particle',"--particle",metavar=None,
            type=str, required=False, help=help_particle)
    argparser.add_argument('-polymer',"--polymer",metavar=None,
            type=str, required=False, help=help_polymer)
    argparser.add_argument('-train',"--train_costum_method",metavar=None,
            type=str, required=False, help=help_train_costum_method)
    argparser.add_argument('-pargs',"--property_args",metavar=None,
            type=str, required=False, help=help_property_args)
    argparser.add_argument('-pkwargs',"--property_kwargs",metavar=None,
            type=str, required=False, help=help_property_kwargs)
    
    parsed_args = argparser.parse_args()
    
    known_init_kwargs = ['memory_demanding','keep_frames','conftype',
                         'adsorption_interval','particle_method','polymer_method',
                        'train_custom_method','polymer','particle']
    
    evaluated_strings = ['trajf','connectivity',
                         'train_costum_method','adsorption_interval',
                         'property_args','property_kwargs']
    margs = dict()
    init_kwargs = dict()
    for s in vars(parsed_args):
        
        attr = getattr( parsed_args,s)
        if attr is None:
            continue
        if s in evaluated_strings:
            try:
                margs[s] = eval(attr)
            except:
                margs[s] = attr
        else:
            margs[s] = attr
        if s in known_init_kwargs:
            init_kwargs[s] = attr
    for k,m in margs.items():
        print(k,type(m),m)
    
    supra = mda.supraClass(margs['topol'],margs['connectivity'],**init_kwargs)
    data = supra.get_property(margs['trajf'],margs['property'],
                              *margs['property_args'],
                              **margs['property_kwargs'])

    return

if __name__=='__main__':
    main()