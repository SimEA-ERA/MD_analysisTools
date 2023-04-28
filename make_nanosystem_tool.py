# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""

import os
import sys
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/MDanalysis')
import md_analysis as mda  
from matplotlib import pyplot as plt
import argparse
import numpy as np

 
def make_dir(path):
    if not os.path.exists(path):
        i = os.system('mkdir ' + path)
        if i == 0:
            print('Created DIR: '+path)
        else:
            os.system('mkdir ' + '\\'.join(path.split('/')))
    return

def get_stoich(obj,t1,t2,nt1,nt2):
    no,na = nty(obj,t1,t2)
    return na/(na+no)*(nt1+nt2)/nt1

def getNumtype(obj,t):
    if mda.ass.iterable(t):
        no = 0
        for ti in t:
            no+=np.count_nonzero(obj.at_types==ti)
    else:
        no = np.count_nonzero(obj.at_types==t)
    return no

def nty(obj,t1,t2):
    
    no = getNumtype(obj,t2)
    na = getNumtype(obj,t1)
    return no,na

def filt_type(obj,ty):
    if mda.ass.iterable(ty):
        filt = False
        for ti in ty:
            filt = np.logical_or(filt,obj.at_types ==ti )
    else:
        filt = obj.at_types == ty
    return filt
    
def number_of_atoms_to_remove(obj,t1,t2,nt1,nt2):
    no,na = nty(obj,t1,t2)
    for n in range(no,0,-1):
        xna = na-nt1*n/nt2
        #print(xna)
        if xna%1 ==0 and xna>0:
            return int(xna),int(no-n)
    return

def raise_Exception(var,varname=''):
    if var is None:
        raise Exception('The variable "{}" is None. Give it a proper value'.format(varname))
    return
class shapes():
    def __init__(self,obj,shape,path=None,prefix='nano',**kwargs):
        self.obj = obj # Analysis object
        self.shape = shape
        for k,v in kwargs.items():
            setattr(self,k,v)
        self.cut_shape_func = getattr(self,'cut_'+shape)
        self.get_surface_filt = getattr(self,'surf_'+shape)
        self.get_surfrm_filt = getattr(self,'surfrm_'+shape)
        self.volume = getattr(self,'volume_'+shape)
        self.kwargs = kwargs
        self.name = prefix + self.shape
        if path is None:
            self.path = '{:s}{:s}'.format(prefix,shape)
        else:
            self.path = path
        return
    
    def get_molname(self):
        x = np.unique(self.obj.mol_names)
        if len(x)>1:
            raise Exception('Found more than one molname in the system')
        else:
            return x[0]
    
    def save_file(self,stabilizing_springs=None):
        path = self.path
        make_dir(path)
        molname = self.get_molname()
        
        self.obj.write_gro_file('{}/{:s}.gro'.format(path,self.name))
        
        
        make_dir('{}/itp'.format(path))
        mda.gromacsTop(self.obj).write_itp(molname, fname='{}/itp/{:s}.itp'.format(path,self.name))
        if stabilizing_springs is not None:
            for k in stabilizing_springs:
                
                mda.gromacsTop(self.obj).write_itp(molname, r=0,k=k,
                        fname='{}/itp/{:s}_k{:d}.itp'.format(self.path,self.name,k))
        return 
    
    def cut(self):
        self.cut_shape_func()
        return
    
    def surf(self,surf_thick):
        filt = self.get_surface_filt(surf_thick)
        return filt
    
    def surfrm(self,surf_thick):
        filt = self.get_surfrm_filt(surf_thick)
        return filt
    
    def cut_sphere(self,**kwargs):
        coords = self.obj.get_coords(0)
        cm = mda.CM(coords,self.obj.atom_mass)
        rcm = coords - cm
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        fsphere = dcm<=self.diameter/2
        self.obj.filter_system(fsphere)
        return
    
    def cut_pore(self):
        coords = self.obj.get_coords(0)
        cm = mda.CM(coords,self.obj.atom_mass)
        rcm = coords[:,0:2] -cm[0:2]
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        fpore = dcm>=self.diameter/2
        rz = np.abs(coords[:,2]-cm[2])
        fl = rz < self.length/2
        
        fpore = np.logical_and(fpore,fl)
        self.obj.filter_system(fpore)
        return
    
    def cut_tube(self):
        coords = self.obj.get_coords(0)
        cm = mda.CM(coords,self.obj.atom_mass)
        rcm = coords[:,0:2] -cm[0:2]
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        finner = dcm>=self.diameter/2
        fouter = dcm<=self.outer_diameter/2
        ftube = np.logical_and(fouter,finner)
        
        rz = np.abs(coords[:,2]-cm[2])
        fl = rz < self.length/2
        
        flntube = np.logical_and(ftube,fl)
        self.obj.filter_system(flntube)
        return
    
    def surf_pore(self,surf_thick):
        coords = self.obj.get_coords(0)
        cm = mda.CM(coords,self.obj.atom_mass)
        rcm = coords[:,0:2] -cm[0:2]
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        fp_s = dcm<=self.diameter/2+surf_thick

        rz = np.abs(coords[:,2]-cm[2])
        fl_s = rz > self.length/2-surf_thick
        s = np.logical_or(fl_s,fp_s)
        return s
    
    def surf_tube(self,surf_thick):
        coords = self.obj.get_coords(0)
        cm = mda.CM(coords,self.obj.atom_mass)
        rcm = coords[:,0:2] -cm[0:2]
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        fin_s = dcm<=self.diameter/2+surf_thick
        fout_s = dcm>=self.outer_diameter/2-surf_thick
        rz = np.abs(coords[:,2]-cm[2])
        fl_s = rz > self.length/2-surf_thick
        s = np.logical_or(fl_s, np.logical_or(fin_s,fout_s))
        return s
    
    def surf_sphere(self,surf_thick):
        coords = self.obj.get_coords(0)
        cm = mda.CM(coords,self.obj.atom_mass)
        rcm = coords -cm 
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        fin_s = dcm>=self.diameter/2-surf_thick
        return fin_s
    
    def surfrm_pore(self,surf_thick):
        coords = self.obj.get_coords(0)
        cm = mda.CM(coords,self.obj.atom_mass)
        rz = np.abs(coords[:,2]-cm[2])
        fl_s = rz > self.length/2-surf_thick
        return fl_s
    
    def surfrm_tube(self,surf_thick):
        coords = self.obj.get_coords(0)
        cm = mda.CM(coords,self.obj.atom_mass)
        rz = np.abs(coords[:,2]-cm[2])
        fl_s = rz > self.length/2-surf_thick
        return fl_s
    
    def surfrm_sphere(self,surf_thick):
        return self.surf_sphere(surf_thick)
    
    
    def volume_pore(self):
        vc = 0.25*np.pi*self.length*self.diameter**2
        box = self.obj.get_box(0)
        return box[0]*box[1]*self.length-vc
    
    def volume_tube(self):
        vc = 0.25*np.pi*self.length*(self.outer_diameter**2 - self.diameter**2)
        return vc
    
    def volume_sphere(self):
        vc = 4/3*np.pi*self.diameter**3
        return vc
    
    def surface_pore(self):
        s1 = np.pi*self.length*self.diameter
        box = self.obj.get_box(0)
        s2 = box[0]*box[1]-0.25*np.pi*self.diameter**2
        return s1 + 2*s2
    
    def surface_tube(self):
                
        s1 = np.pi*self.length*self.diameter
        s2 = np.pi*self.length*self.outer_diameter
        s3 = 0.25*np.pi*(self.outer_diameter**2-self.diameter**2)
        return s1+s2+2*s3
    
    def surface_sphere(self):
        s = 4*np.pi*self.diameter**2
        return s
    
    
def make_stoichiometric(obj,shape,type1,type2,ntype1,ntype2,surf_thick):
        raise_Exception(type1,'type1')
        raise_Exception(type2,'type2')
        raise_Exception(ntype1,'ntype1')
        raise_Exception(ntype2,'ntype2')
        print('Making the system stoichiometric ... ')
        stoich = get_stoich(obj, type1, type2, ntype1, ntype2)
        print('cut {:s} stoich = {:4.8f}'.format(shape.name,stoich))
        
        # Atoms to remove for making stoichiometric
        print('non-stoichiometric n{:},n{:} = {:}'.format(type1,type2,nty(obj,type1,type2)))
        na,no = number_of_atoms_to_remove(obj,type1,type2,ntype1,ntype2)
        
        print('removing {:d} {:} and {:d}  {:}'.format(na,type1,no,type2))
        fsurf = shape.surfrm(surf_thick)
        fsurf1 = np.logical_and(fsurf,filt_type(obj,type1))
        fsurf2 = np.logical_and(fsurf,filt_type(obj,type2))
        nsurf1 = np.count_nonzero(fsurf1)
        nsurf2 = np.count_nonzero(fsurf2)
        print('surface {:} = {:d}, surface {:} = {:d}'.format(type1,nsurf1,type2,nsurf2))
        if na > nsurf1:
            raise Exception('surface {} < {} to remove'.format(type1,type1))
        if no > nsurf2:
            raise Exception('surface {} < {} to remove'.format(type2,type2))
        
        #Performing the removal
        args1 = obj.at_ids[fsurf1]
        args1 = np.random.choice(args1.copy(),na,replace=False)
        args2 = obj.at_ids[fsurf2]
        args2 = np.random.choice(args2.copy(),no,replace=False)
        
        args = list(np.concatenate((args1,args2),dtype=int))
       
        filt = np.ones(obj.natoms,dtype=bool)
        for i in range(obj.natoms):
            if i in args:
                filt[i] = False
                
        assert  obj.natoms - np.count_nonzero(filt) == args1.shape[0] + args2.shape[0],'number of args to be removed is not equal to False in filter'
        
        obj.filter_system(filt)
        print('new natoms {},{} = {}'.format(type1,type2,nty(obj,type1,type2)))
        print('stoichiometric {:s} stoich = {:4.8f}'.format(shape.name,get_stoich(obj,type1,type2,ntype1,ntype2)))
        return
    
def change_surface_types(surface_types_map,obj,shape,surf_thick,
                         new_masses=dict(),new_charges=dict()):
    for key,val in surface_types_map.items():    
        fch = np.logical_and(shape.surf(surf_thick),filt_type(obj, key))
        for i in range(obj.natoms):
            if fch[i]: 
                obj.at_types[i] = val
        newty_filt = filt_type(obj,val)
        nfch = np.count_nonzero(newty_filt)
        if val not in new_masses:
            obj.mass_map[val] = obj.mass_map[key] 
        if val not in new_charges:
            obj.charge_map[val] = obj.charge_map[key] 
        print('n{:s} = {:d} , '.format(val,nfch))
    return
    
def wrap_surface(fname,aluitp,diameter=None,length=None,
                    write=False,nocut=False,size=None,
                    stabilizing_springs=None,offset=[0,0,0],
                    surf_thick=0.3, 
                    type1=None, type2=None,
                    ntype1=None, ntype2=None,
                    surface_types_map=None):
    def norm2(r):
        return np.sqrt(np.dot(r,r))
    obj = mda.Analysis(fname,aluitp)
    obj.read_gro_file()
    # find dimensions
    c = obj.get_coords(0)
    thickness = c[:,2].max()-c[:,2].min()
    surf_size = obj.get_coords(0).max(axis=0) - obj.get_coords(0).min(axis=0)
    # find how to multiply
    mult_1 = int((np.pi*diameter+thickness/2)/surf_size[1])
    mult_0 = int(length/surf_size[0])
    
    multiplicity = (mult_0,mult_1,0)
        
    obj.multiply_periodic(multiplicity)
    diff = obj.get_coords(0) - obj.get_coords(0).min(axis=0)
    if nocut==False:
        fl = diff[:,0] <length
        fw = diff[:,1] < np.pi*diameter+thickness/2
        obj.filter_system(np.logical_and(fl,fw))
        box = obj.get_coords(0).max(axis=0) - obj.get_coords(0).min(axis=0)
        obj.timeframes[0]['boxsize'] = box
    else:
        box = obj.get_box(0)

    
    # wrap_the surface
    coords = obj.get_coords(0)
    thickness = coords[:,2].max()-coords[:,2].min()


    cm = mda.CM(coords,obj.atom_mass)
    cm_init = cm.copy()
    coords-=cm
    L = box[1]
    
    wrapped_coords = np.empty_like(coords)
    
    cm = mda.CM(coords,obj.atom_mass)
    rz = cm.copy() ; rz[2] -= L/(4*np.pi)
    d = norm2(rz)
    theta = []
    for i in range(coords.shape[0]):
        rold = coords[i] - rz
    
        th = (rold[1]-rz[1])*2*np.pi/L
        theta.append(th)

        rr = rold-cm
        rrot = np.array( [rold[0],(d+rr[2])*np.cos(th),(d+rr[2])*np.sin(th)] )
        
        wrapped_coords[i] = rrot
    
    
    wc = wrapped_coords.copy()
    wc[:,0]=wrapped_coords[:,2]
    wc[:,2] = wrapped_coords[:,0]
    wrapped_coords = wc
    wrappedtube_size = wrapped_coords.max(axis=0) - wrapped_coords.min(axis=0)
    offset = np.array(offset)
    
    box  = [size,size,size] +offset
    obj.timeframes[0]['boxsize'] = box
    
    alu_cm = mda.CM(obj.get_coords(0),obj.atom_mass)
    #wrapped_coords+=box/2
    obj.timeframes[0]['coords'] = wrapped_coords

    obj.timeframes[0]['coords'] += obj.get_box(0)/2 -alu_cm

    final_coords = obj.get_coords(0)
    cm = mda.CM(final_coords,obj.atom_mass)
    fcm = final_coords[:,0:2] - cm[0:2]
    inner_diameter = 2*np.sqrt(np.sum(fcm*fcm,axis=1)).min()
    outer_diameter = 2*np.sqrt(np.sum(fcm*fcm,axis=1)).max()
    length = wrappedtube_size[2]
    for k,s in zip([inner_diameter,outer_diameter,length],['inner_diameter','outer_diameter','length']):
        print('final {:s} = {:4.3f}'.format(s,k))
    
    if inner_diameter > size:
        raise Exception('Final inner_diameter = {:4.3f} is more than the box size, increase the box size or reduce diameter'.format(inner_diameter))
    elif outer_diameter > size:
        raise Exception('Final outer_diameter = {:4.3f} is more than the box size, increase the box size or reduce diameter'.format(outer_diameter))
    elif length > size:
        raise Exception('Final length = {:4.3f} is more than the box size, increase the box size or reduce length'.format(length))
    
    shape = shapes(obj,'tube',prefix='wrapped',
                   diameter=inner_diameter,
                   outer_diameter=outer_diameter,
                   length=length)
    if nocut==False:
        make_stoichiometric(obj,shape,type1,type2,ntype1,ntype2,surf_thick)
    if surface_types_map is not None:
        change_surface_types(surface_types_map,obj,shape,surf_thick)
    
    obj.mol_ids = np.ones(obj.natoms,dtype=int)
    obj.analysis_initialization()
    
    if write:
        shape.save_file(stabilizing_springs=stabilizing_springs)

    return obj,shape

def make_nano(nanoshape,gro_file,itp_file,
                    xdim=10,ydim=10,zdim=10,write=True, boxoffset=[0,0,0],
                    make_tetr=True,make_stoich=True,type1=None, type2=None,
                    ntype1=None, ntype2=None, surface_types_map = dict(),
                    surf_thick=0.3, new_masses = dict(), stabilizing_springs=None,
                    new_charges=dict(),**shape_kwargs):
    #global obj
    obj = mda.Analysis(gro_file,itp_file)
    
    obj.read_file()
    box = obj.get_box(0)
    
    multiplicity=(int(xdim/box[0]),int(ydim/box[1]),int(zdim/box[2])) 

    obj.multiply_periodic(multiplicity)

    coords = obj.get_coords(0)

    
    fx = np.logical_and(coords[:,0]>=0,coords[:,0]<xdim)
    fy = np.logical_and(coords[:,1]>=0,coords[:,1]<ydim)
    fz = np.logical_and(coords[:,2]>=0,coords[:,2]<zdim)
    obj.filter_system(np.logical_and(np.logical_and(fx,fy),fz))

    shape = shapes(obj,nanoshape,**shape_kwargs)
    shape.cut()
   
    
    
    obj.timeframes[0]['boxsize']= np.array([xdim,ydim,zdim])

    if make_stoich:
        make_stoichiometric(obj,shape,type1,type2,ntype1,ntype2,surf_thick)
    
    obj.timeframes[0]['boxsize'] = np.array([xdim,ydim,zdim])+np.array(boxoffset)
    coords = obj.get_coords(0) 
    obj.timeframes[0]['coords']+= obj.get_box(0)/2 - mda.CM(coords,obj.atom_mass)
    
    if surface_types_map is not None:
        change_surface_types(surface_types_map,obj,shape,surf_thick)
        
    print('system size = {:d}'.format(obj.natoms))
    obj.mol_ids = np.ones(obj.natoms,dtype=int)
    obj.analysis_initialization()
    
    if write: 
        shape.save_file(stabilizing_springs)
    
    obj.timeframes[0]['coords'] += obj.get_box(0)/2-mda.CM(obj.get_coords(0),obj.atom_mass)
    return obj,shape

def make_bulk_fit_inbox(initbulk,fitp,size):
    #global traj
    
    traj = mda.Analysis(initbulk,fitp)
    traj.read_file()
    frame = list(traj.timeframes.keys())[0]
    box = traj.get_box(frame)
    multiplicity = tuple(int(size[i]/box[i]) for i in range(3))
    traj.multiply_periodic(multiplicity)
    
    new_box = traj.get_box(frame)
    
    def crit(c):
        cri = False
        for i in range(3):
            cri = np.logical_or(cri,c[:,i]<0)
            cri = np.logical_or(cri,c[:,i]>size[i])
        return cri
    traj.remove_residues(crit)
    traj.timeframes[0]['boxsize'] = np.array([s for s in size])

    return traj,traj.nmolecules


def mchains_inbox(M,initbulk,fitp,size,onlyz=False,
                  maxtrials=20,scaleFactor=1.15):
    box = np.array([size,size,size])
    if onlyz:
        scaleFactor=1.8
    tr,nm = make_bulk_fit_inbox(initbulk,fitp,box)
    sols = [nm]
    zs = [size]
    
    j=0
    stop =False
    while M!=nm and j< maxtrials: 
        
        s = np.array(sols)
        zsa = np.array(zs)
        f1 = s<M
        f2 = s>M
        l1 = f1.any()
        l2 = f2.any()
        if l1 and l2:
            z1 = zsa[f1].max()
            z2 = zsa[f2].min()
            size = (z2+z1)/2
            diff = z2-z1
                
            if diff<1e-3:
                print('iteration {:d}  size -->{:4.8f} , diff = {:4.8}'.format(j,size,diff))
                stop = True
        elif l1 and not l2:
            
            size*=scaleFactor
        elif l2 and not l1:
            size/=scaleFactor
        
        mnold = nm
        
        if stop or j == maxtrials:
            size = z2
        if onlyz:
            box[2] = size
        else:
            box = np.array([size,size,size])
        tr,nm = make_bulk_fit_inbox(initbulk, fitp, box)
                    
        sols.append(nm)
        zs.append(size)
        
        print('Iteration {:d} Number of residues = {:d}'.format(j,nm))
        if j==maxtrials or stop:
            print('Removing randomly {:d} residues'.format(nm-M))
            ids = np.unique(tr.mol_ids)
            idsr = np.random.choice(ids,M,replace=False)
            filt = np.isin(tr.mol_ids,idsr)
            tr.filter_system(filt)
            break
        j+=1
    
    traj = tr
    return traj,traj.nmolecules
    
    

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

def get_stoichtypes(argstring):
    key1,key2 = argstring.split(',')
    
    def tn(key):
        ty,nt = key.strip().split(':')
        ty = ty.strip().split('-')
        nt = int(nt)
        return ty,nt
    type1,ntype1 = tn(key1)
    type2,ntype2 = tn(key2)
    return type1,type2,ntype1,ntype2
def get_defaults(command_args):
    if command_args.boxoffset ==0:
        command_args.boxoffset=[0.0,0.0,0.0]
    
    type1,type2,ntype1,ntype2 = get_stoichtypes(command_args.types)
    size = command_args.box
    box = [size,size,size]
    
    
    if command_args.stabilizing_springs is None:
        stabilizing_springs = [k for k in range(5000,500,-1000)]   
        stabilizing_springs += [k for k in range(500,100,-100)] 
        stabilizing_springs += [k for k in range(50,10,-10)]
        stabilizing_springs += [k for k in range(5,0,-1)]
    else:
        stabilizing_springs = command_args.stabilizing_springs
    if (np.array(stabilizing_springs)<0).any():
        stabilizing_springs = None
        
    shape = command_args.shape
    
    if shape=='tube':
        ka = ['diameter','length','outer_diameter']
        onlyz=False
    elif shape == 'pore':
        onlyz=True
        ka = ['diameter','length']
    elif shape == 'sphere':
        ka = ['diameter']
        onlyz=False
    elif shape=='wrap':
        onlyz=False
        ka = ['diameter','length']
    elif shape =='asis':
        ka =  dict()
        onlyz=True
    kargs = dict()
    for k in ka:
        kargs[k] = getattr(command_args,k)
    
    return shape, size, box, type1, type2, ntype1, ntype2, stabilizing_springs, kargs, onlyz

def particle_Checks(command_args):
   
    if command_args.shape =='asis':
        return
   
    if command_args.box is None:
        raise Exception('Box (-b) is required')
    else:
        size = command_args.box
   
    if command_args.shape in ['pore','tube','wrap','sphere']:    
        if command_args.diameter is None:
            raise Exception('diameter (-d) is required')
   
    if command_args.shape in ['pore','tube','wrap']:    
        if command_args.length is None:
            raise Exception('length (-l) is required')
     
    if command_args.shape in ['tube']:    
        if command_args.outer_diameter is None:
            raise Exception('outer diameter (-od) is required')
            
    if command_args.diameter > size:
        raise Exception('diameter is larger than the size')

    if command_args.shape in ['pore','tube'] and command_args.length  > size:
        raise Exception('length is larger than the size')

    if command_args.shape =='pore' and command_args.diameter > size-0.5:
        raise Exception('Pore is too thin. Decrease diameter (-d) or increase box (-b)')

    if command_args.volume_fraction is not None and not 0<command_args.volume_fraction<1:
        raise Exception('volume fraction (-vf) must be between (0,1)')
   
    if command_args.shape=='tube':   
        od = command_args.outer_diameter
        if  od is None:
            raise Exception('give outer_diameter(-od) of nanotube')
        elif od > size:
            raise Exception('outer_diameter(-od) is greater than the size')
    
    return 

def main():
    adddef = " [  default: %(default)s ]"
    argparser = argparse.ArgumentParser(description="Make a Nanosystem using bulk solid and bulk polymer material")
    
    argparser.add_argument('-fs',"--fsolid",metavar=None,
            type=str, required=True, 
            help="gro file of equilibrated bulk solid material. This bulk material will be cutted in the specified shape or be wrapped in cylindrical shape. Use a proper surface when wrapping")
    argparser.add_argument('-itps',"--itpsolid",metavar=None,
            type=str, required=True, 
            help="itp file of solid material. Needed to get the bond information and types")
    
    argparser.add_argument('-fp',"--fpolymer",metavar=None,
            type=str, required=False, 
            help="gro file of  polymer material")
    
    argparser.add_argument('-itpp',"--itppolymer",
            type=str, required=False, metavar=None,
            help="itp file of polymer material. Needed to get the bond information and types")
    
    argparser.add_argument('-s',"--shape",metavar=None,
            type=str, required=True, 
            help="Geometry of cutting the solid material",
            choices=['tube','sphere','pore','wrap','asis'])
    
    argparser.add_argument('-b',"--box",metavar=None,
            type=float, required=False, 
            help="Initial size of the system. xy dimensions for pore")
    
    argparser.add_argument('-d',"--diameter",
            type=float, required=False, metavar=None,
            help="diameter of the geometry. For tube is inner")
    
    argparser.add_argument('-od',"--outer_diameter",
            type=float, required=False, metavar=None,
            help="diameter of the geometry. For tube is inner. For wrapped tube is mean")
    
    argparser.add_argument('-l',"--length",
            type=float, required=False, metavar=None,
            help="length of pore or nanotube")
    
    argparser.add_argument('-o',"-outputfile",metavar=None,
            type=str, default = 'merged.gro', 
            help="name of the output file"+adddef)
    
    argparser.add_argument('-nc',"--nchains",metavar=None,
            type=int, required=False, 
            help='Number of chains. Specify either this explicitly or volume fraction (vf) and denisty per chain (prho)')
    
    argparser.add_argument('-vf',"--volume_fraction",
            type=float, required=False, metavar=None,
            help='Volume fraction of the nanoparticle based on bulk polymer properties')
    
    argparser.add_argument('-mw',"--molweight",metavar=None,
            type=float, default = 1687.32, 
            help='Chain molecular weight'+adddef)
    
    argparser.add_argument('-prho',"--polymerrho",metavar=None,
            type=float, default = 0.847, 
            help='bulk polymer density'+adddef)
    
    argparser.add_argument('-bt',"--bulk_translation",metavar=None,
            type=float, default = 1.0, 
            help='minimum "z" distance of solid and polymer'+adddef)
    
    argparser.add_argument('-top',"--top",metavar=None,
            type=str, required=False, 
            help='.top file to write the new number of chains')
    
    argparser.add_argument("-nocut","--nocut", metavar=None,
                           type=bool,default=False, 
                           help="Valid when shape is 'wrap' . It does not cut the wrapped surface. Useful to have 'perfect' wrapped surface"+adddef)
    
    argparser.add_argument("-stoich","--stoichiometry", metavar=None,
                           type=bool,default=True, 
                           help="Make the surface stoichiometric"+adddef)
    
    argparser.add_argument("-sth","--surface_thickness", metavar=None,
                           type=float,default=0.15, 
                           help="surface_thickness"+adddef)
    argparser.add_argument("-k","--stabilizing_springs", metavar=None,
                           type=int,nargs='+', required=False,
                           help="spring constants for stabilization of the new nanoparticle. Pass -1 for no stabilizing springs")
    argparser.add_argument("-t","--types", metavar=None,
                           type=str, default='Alt-Alo:2,O:3',
                           help="types for stoichiometric calculations "+adddef)
    argparser.add_argument("-boxof","--boxoffset", metavar=None,
                           type=float,nargs='+', required=False,default=0,
                           help="boxoffset"+adddef)
    

    
    command_args = argparser.parse_args()
        

    shape, size, box, type1, type2, ntype1, ntype2, stabilizing_springs, kargs, onlyz = get_defaults(command_args)
    
    particle_Checks(command_args)
    
    if shape == 'asis':
        alu = mda.Analysis(command_args.fsolid,command_args.itpsolid)
        alu.read_file()
        size = alu.get_box(0)[0]
        
    elif shape!='wrap':
        alu,cs = make_nano(shape,command_args.fsolid,command_args.itpsolid, 
                xdim=size, ydim = size, zdim=size,
                type1=type1,
                type2=type2,
                ntype1=ntype1,
                ntype2=ntype2,
                surf_thick=command_args.surface_thickness,
                make_stoich=command_args.stoichiometry,
                surface_types_map={'Alo':'Alt'},
                stabilizing_springs= stabilizing_springs,
               boxoffset=command_args.boxoffset,
                **kargs)
    else:
        
        alu,cs = wrap_surface(command_args.fsolid,command_args.itpsolid,
                    write=True,
                    nocut=command_args.nocut,
                    size = command_args.box,
                    type1=type1,
                    type2=type2,
                    ntype1=ntype1,
                    ntype2=ntype2,
                    surface_types_map=None,
                    stabilizing_springs=stabilizing_springs,
                    surf_thick=command_args.surface_thickness,
                    offset=command_args.boxoffset,
                    **kargs)
    
    print('Done making particle')
    if command_args.fpolymer is None and command_args.itppolymer is None:
        exit()
    elif command_args.fpolymer is None and command_args.itppolymer is not None:
        raise Exception('Give initial polymer gro file (-fp)') 
    elif command_args.fpolymer is not None and command_args.itppolymer is None:
        raise Exception('Give polymer itp file (-itpp)') 

    if command_args.nchains is None  and command_args.volume_fraction is None:
            raise Exception('Give either number of chains (-nc) or volume fraction (-vf). In case of -vf option Remember the default molecular weight (-mw) and the default bulk density(-prho) are suitable for 30-mer cis-1,4 PB at 413K')   
    
    print('Merging with polymer system ...')
    if command_args.nchains is not None:
        M = command_args.nchains
        if M <0:
            
            vp = command_args.box**3
            brho =command_args.polymerrho
            mch = command_args.molweight*1.660539e-3
            Mp = vp/(mch*brho)
            M = int(round(Mp,0))
            print('setting the number of chains automatically to {:d}'.format(M))
    elif  command_args.volume_fraction is not None:
        vf = command_args.volume_fraction
        vs = cs.volume()
        vp = vs*(1-vf)/vf
        brho =command_args.polymerrho
        mch = command_args.molweight*1.660539e-3
        Mp = vp/(mch*brho)
        M = int(round(Mp,0))

    
    bulk,nm = mchains_inbox(M,command_args.fpolymer,
                            command_args.itppolymer,size,
                            maxtrials=30,onlyz=onlyz)
    
    
    merged = merge_n_translate('merged.gro',alu,bulk,
                               bulk_translation=command_args.bulk_translation)
    
    if command_args.top is not None:
        ftop = command_args.top
        with open(ftop,'r') as f:
            lines = f.readlines()
            bname = bulk.mol_names[0]
            for i,line in enumerate(lines):
                
                if bname == line[:2]:
                    lines[i] ='{:s}  {:d} \n'.format(bname,bulk.nmolecules)
        f.closed

        with open(ftop,'w') as f:
            for line in lines:
                f.write(line)
        f.closed

    print('Done Merging')
if __name__ =='__main__':
    main()


