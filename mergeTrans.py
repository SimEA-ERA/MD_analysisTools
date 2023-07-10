import numpy as np
import md_analysis as mda
grofile1 = '.gro' ; itp1 = ['.itp','.itp']
grofile2='.gro'   ; itp2= ['.tip']
system1 = mda.Analysis(grofile1,itp1)
system2 = mda.Analysis(grofile2,itp2)

def  merge_n_translate(output_name,alu,bulk,bulk_translation=1.0):
    import copy
    
    
    
    boxa = alu.get_box(0)
    boxb = bulk.get_box(0)
    box =  np.maximum(boxa,boxb)
    
    
    alucm = mda.CM(alu.get_coords(0),alu.atom_mass)
    bulkcm = mda.CM(bulk.get_coords(0),bulk.atom_mass)
    bulk.timeframes[0]['coords']+=box/2-bulkcm
    alu.timeframes[0]['coords']+=box/2-alucm
    alucm = mda.CM(alu.get_coords(0),alu.atom_mass)
    
    merged = copy.deepcopy(bulk)
    merged.timeframes[0]['coords'][:,2]+=-bulk.get_coords(0)[:,2].min()+alu.get_coords(0)[:,2].max()+bulk_translation
    
    
    merged.merge_system(alu)
    box[2] = merged.get_coords(0)[:,2].max()-merged.get_coords(0)[:,2].min()+bulk_translation
    merged.timeframes[0]['boxsize'] = box
    
    merged.timeframes[0]['coords'][:,2] += box[2]/2-alucm[2]
    merged.write_gro_file('{:s}'.format(output_name))
    return bulk

merged = merge_n_translate('merged.gro', system1, system2,bulk_translation=1.0)