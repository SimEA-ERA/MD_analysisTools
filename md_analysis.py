import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
import time
from time import perf_counter
from scipy.optimize import minimize,dual_annealing,fsolve,differential_evolution
from numba import jit,njit,prange
from numba.experimental import jitclass
import warnings
import inspect
import collections
import six
from pytrr import GroTrrReader
import pytrr
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import logging
import coloredlogs

LOGGING_LEVEL = logging.DEBUG

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
logFormat = '%(asctime)s\n[ %(levelname)s ]\n[%(filename)s -> %(funcName)s() -> line %(lineno)s]\n%(message)s\n --------'
formatter = logging.Formatter(logFormat)

logfile_handler = logging.FileHandler('md_analysis.log',mode='w')
logfile_handler.setFormatter(formatter)

logger.addHandler(logfile_handler)

stream = logging.StreamHandler()
stream.setLevel(logging.DEBUG)
stream.setFormatter(formatter)

logger.addHandler(stream)

fieldstyle = {'asctime': {'color': 'magenta'},
              'levelname': {'bold': True, 'color': 'green'},
              'filename':{'color':'green'},
              'funcName':{'color':'green'},
              'lineno':{'color':'green'}}
                                   
levelstyles = {'critical': {'bold': True, 'color': 'red'},
               'debug': {'color': 'blue'}, 
               'error': {'color': 'red'}, 
               'info': {'color':'cyan'},
               'warning': {'color': 'yellow'}}

coloredlogs.install(level=LOGGING_LEVEL,
                    logger=logger,
                    fmt=logFormat,
                    datefmt='%H:%M:%S',
                    field_styles=fieldstyle,
                    level_styles=levelstyles)



def stay_True(dic):
    keys = list(dic.keys())
    stayTrue = {keys[0]:dic[keys[0]]}
    for i in range(1,len(dic)):
        stayTrue[keys[i]] = np.logical_and(stayTrue[keys[i-1]],dic[keys[i]])
    return stayTrue

def become_False(dic):
    keys = list(dic.keys())
    bFalse = {keys[0]:dic[keys[0]]}
    for i in range(1,len(dic)):
        bFalse[keys[i]] = np.logical_and(bFalse[keys[0]],np.logical_not(dic[keys[i]]))
    return bFalse


def iterable(arg):
    return (
        isinstance(arg, collections.Iterable) 
        and not isinstance(arg, six.string_types)
    )
import logging
def print_time(tf,name,nf=None):
    s1 = readable_time(tf)
    if nf is None:
        s2 =''
    else:
        s2 = ' Time/frame --> {:s}\n'.format( readable_time(tf/nf))
    x = '-'*(len(name)+11)
    logger.info('Function "{:s}"\n{:s} Total time --> {:s}'.format(name,s2,s1))
def readable_time(tf):
    hours = int(tf/3600)
    minutes = int((tf-3600*hours)/60)
    sec = tf-3600*hours - 60*minutes
    dec = sec - int(sec)
    sec = int(sec)
    return '{:d}h : {:d}\' : {:d}" : {:0.3f}"\''.format(hours,minutes,sec,dec)
        
@jit(nopython=True,fastmath=True)
def distance_kernel(d,coords,c):
    for i in range(d.shape[0]):
        d[i] = 0
        for j in range(3):
            rel = coords[i][j]-c[j]
            d[i] += rel*rel
        d[i] = d[i]**0.5
    return 
@jit(nopython=True,fastmath=True)
def smaller_distance_kernel(d1,d2,c1,c2):
    for i in range(c1.shape[0]):
        distance_kernel(d2,c2,c1[i])
        d1[i] = 10000000000
        for j in range(d2.shape[0]):
            if d2[j]<d1[i]: 
                d1[i] = d2[j]
    return

@jit(nopython=True,fastmath=True)
def running_average(X,every=1):
    n = X.shape[0]
    xrun_mean = np.zeros(n)
    for j in range(0,len(X),every):
        y = X[:j+1]
        n = y.shape[0]
        xrun_mean[j] = np.sum(y)/n
    return xrun_mean

def moving_average(a, n=10) :
    mov = np.empty_like(a)
    
    n2 = int(n/2)
    if n2%2 ==1: n2+=1
    up = a.shape[0]-n2
    for i in range(n2):
        mov[i] = a[:2*i+1].mean()
    for i in range(n2,up):
        mov[i] = a[i-n2:i+n2+1].mean()
    for i in range(up,a.shape[0]):
        j = (a.shape[0]-i)-1
        mov[i] = a[up-j:].mean()
        
    return mov

def block_average(a, n=100) :
    bv = np.empty(int(a.shape[0]/n),dtype=float)
    for i in range(bn.shape[0]):
        x = a[i*n:(i+1)*n]
        bv[i] = x.mean()
    return bv

def block_std(a, n=100) :
    bstd = np.empty(int(a.shape[0]/n),dtype=float)
    for i in range(bn.shape[0]):
        x = a[i*n:(i+1)*n]
        bstd[i] = x.std()
    return bstd


class Energetic_Analysis():
    
    def __init__(self,file):
        self.data_file = file
        if self.data_file[-4:] =='.xvg':
            self.xvg_reader()
        else:
            raise NotImplementedError('Currently only accepting xvg files')
    
    def xvg_reader(self):
        
        with open(self.data_file) as f:
            lines = f.readlines()
            f.closed
            
        columns = ['time']
        for i,line in enumerate(lines):
            l = line.split()
            
            if l[0]=='@' and l[1][0]=='s' and l[2] =='legend':
                columns.append(l[3].strip('"'))
                last_legend = i
            elif line[0] =='@' or line[0]=='#':
                continue
            else:
                break
            
        data = np.array([line.split() for line in lines[last_legend+1:]],dtype=float)
        self.data = pd.DataFrame(data,columns=columns)
        return
    def simple_plot(ycols,xcol='time',size = 3.5,dpi = 300, 
             title='',func=None,
             xlabel=['time (ps)'],save_figs=False,fname=None,path=None):
        figsize = (size,size)
        fig =plt.figure(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=1.5*size)
        plt.tick_params(direction='in', which='major',length=1.5*size)
        plt.xlabel('time (ps)',fontsize=size*3)
        x = self.data[xcol].to_numpy()
        if func is not None:
            funcs = globals()[func]
        for ycol in ycols:
            y = self.data[ycol].to_numpy()
            plt.plot(x,y,lab=ycol)
            if funcs is not None:
                plt.plot(x,funcs(y),lab ='{} - {}'.format(ycol,func))
        plt.legend(frameon=False)
        if save_figs:plt.savefig('{}\{}'.format(path,fname),bbox_inches='tight')
        plt.show()

class Analytical_Expressions():
    @staticmethod
    def KWW(t,A,tc,beta,tww):
        #Kohlrausch–Williams–Watts
        phi = A*np.exp( ( -(t-tc)/tww )**beta )
        return phi
    @staticmethod
    def expDecay(t,A,t0):
        phi = 1+A*( np.exp(-t/t0) - 1 )
        return phi
    @staticmethod
    def expDecay_KWW(t,A1,A2,tc,t0,beta,tww):
        tl =  t[t<tc]
        tup = t[t>=tc]
        phil = Analytical_Expressions.expDecay(tl,A1,t0)
        A2 = Analytical_Expressions.expDecay(tc,A1,t0) #continuity
        phiup = Analytical_Expressions.KWW(tup,A2,tc,beta,tww)
        return np.concatenate( (phil,phiup) )
    
@jitclass
class Analytical_Functions():
    def __init__(self):
        pass

    @staticmethod
    def Rz(th):
        R = np.array([[1, 0 ,0], 
            [0, np.cos(th), -np.sin(th)], 
            [0, np.sin(th), np.cos(th)]])
        return R
    @staticmethod
    def Ry(th):
        R = np.array([[np.cos(th), 0, np.sin(th)] ,
                       [0, 1, 0], 
                       [-np.sin(th), 0, np.cos(th)]])
        return R
    @staticmethod
    def Rx(th):
        R = np.array([[np.cos(th), -np.sin(th), 0],
                     [np.sin(th), np.cos(th), 0],
                     [0,0,1]])
        return R
        
 
    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def q_mult(q1,q2):
        w1 = q1[0] ; x1 = q1[1] ; y1 = q1[2] ; z1 = q1[3]
        w2 = q2[0] ; x2 = q2[1] ; y2 = q2[2] ; z2 = q2[3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return np.array((w, x, y, z))
    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def quaternionConjugate(q):
        w = q[0] ; x = q[1] ; y = q[2] ; z = q[3]
        return np.array((w, -x, -y, -z))
    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def rotate_around_an_axis(axis,r,theta):
        q1 = (0,r[0],r[1],r[2])
        th2 = 0.5*theta
        c = np.cos(th2)
        s = np.sin(th2)
        
        q2 = np.array((c,s*axis[0],s*axis[1],s*axis[2]))
        
        q2c = Analytical_Functions().quaternionConjugate(q2)
        q = Analytical_Functions().q_mult(q1,q2c)
        q3 = Analytical_Functions().q_mult(q2,q)
        return np.array((q3[1],q3[2],q3[3]))
    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def rotate_to_theta_target(rp,r0,rrel,theta_target):
        th0 = calc_angle(rp,r0,r0+rrel)
        theta = theta_target - th0
        naxis = np.cross((rp-r0)/norm2(rp-r0),rrel/norm2(rrel))
        rrot = Analytical_Functions().rotate_around_an_axis(naxis, rrel, theta)
        newth = calc_angle(rp,r0,r0+rrot)
        #logger.debug('th0 = {:6.5f}, new th = {:6.5f} '.format(th0*180/np.pi,newth*180/np.pi))
        return rrot,newth
    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def position_hydrogen_analytically_endgroup(bondl,theta,r1,r0,nh=1):
        r01 = r1-r0
        ru01 = r01/norm2(r01)
        rp = r0 + bondl*ru01
        dhalf = bondl/np.sqrt(3)
        s = -np.sign(r01)
        
        rrel = np.array([s[0]*dhalf,s[1]*dhalf,-s[2]*dhalf])
        #logger.debug('theta target = {:4.3f}'.format(theta*180/np.pi))
        
        newth = theta +1    
        af = Analytical_Functions()
        while np.abs(newth - theta)>1.e-4:
            rrel,newth = af.rotate_to_theta_target(rp, r0, rrel, theta)
        
        r2 = r0 +rrel
       
        if nh == 1 :
            return r2
            
        rn = af.position_hydrogen_analytically(bondl,theta,rp,r0,r2,nh-1)

        return rn
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def position_hydrogen_analytically_cis(bondl,theta,r1,r0,r2,nh=1):
        
        r01 = r1-r0
        r02 = r2-r0
        r1 = r0 +bondl*(r01)/norm2(r01)
        r2 = r0 +bondl*(r02)/norm2(r02)
        
        rm = 0.5*(r1+r2)
        ru1 = r2 - r1 ; u1 = ru1/norm2(ru1)
        ru2 = r0 - rm ; u2 = ru2/norm2(ru2)
        rn = r0 + bondl*u2
        return rn
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def position_hydrogen_analytically(bondl_h,theta,r1,r0,r2,nh=1):
        '''
        Parameters
        ----------
        Works for CH2 groups, can be used for CH3 if one hydrogen is added proberly
        bondl_h :Float (length units)
            length of the hydrogen bond.
        theta : float (radians)
            angle between r1-r0-newH or r2-r0-newH. It should be equivalent 
        r1 : float with shape (3,)
            coordinates atom to the left
        r0 : float with shape (3,)
            coordinates atom to the middle
        r2 : float with shape (3,)
            coordinates atom to the right
        nh : int 
            Solution 1 or 2
            The default is 1.
    
        Returns
        -------
        rh : float with shape (3,)
            coordinates of added atom
    
        '''
        r01 = r1-r0
        r02 = r2-r0
        r1 = r0 +bondl_h*(r01)/norm2(r01)
        r2 = r0 +bondl_h*(r02)/norm2(r02)
        
        rm = 0.5*(r1+r2)
        
        ru1 = r2 - r1 ; u1 = ru1/norm2(ru1)
        ru2 = r0 - rm ; u2 = ru2/norm2(ru2)
        
        u3 = np.cross(u1,u2)
        a = theta/2
        if nh ==1:
            rh = r0 + bondl_h*(np.cos(a)*u2 + np.sin(a)*u3)
        elif nh ==2:
            rh = r0 + bondl_h*(np.cos(a)*u2 - np.sin(a)*u3)
        return rh

class add_atoms():
    def __init__(self):
        pass
    
    hydrogen_map = {'CD':[1,0.11,116,(1,),'_cis'],
                    'C':[2,0.11,109.47,(1,2),''],
                    'CE':[3,0.109,109.47,(1,2,3),'_endgroup']}
    
    
    @staticmethod
    def append_atoms(self,k='ghost'):
        t0 = perf_counter()
        for s in ['at_ids','at_types','mol_names','mol_ids']:
            a1 = getattr( self, s )
            a2 = getattr( self,'_'.join([k,s]) )
            n12 = np.concatenate((a1,a2))
            setattr(self,s,n12)
        for frame in self.timeframes:
            c1 = self.timeframes[frame]['coords']
            c2 = self.timeframes[frame]['_'.join((k,'coords'))]
            c12 = np.concatenate((c1,c2))
            self.timeframes[frame]['coords'] = c12
        
        self.connectivity.update(self.ghost_connectivity)
        self.analysis_initialization()
        
        if self.__class__.__name__ == 'Analysis_Confined':
            self.confined_system_initialization()
        print_time(perf_counter()-t0,
                   inspect.currentframe().f_code.co_name,frame+1)
        return 
    
    @staticmethod
    def add_ghost_hydrogens(self,types,noise=None):
        t0 = perf_counter()
        
        new_atoms_info = add_atoms.get_new_atoms_info(self,'h',types)
        
        self.ghost_atoms_info = new_atoms_info
        
        add_atoms.set_ghost_connectivity(self, new_atoms_info)
        
        add_atoms.assign_ghost_topol(self, new_atoms_info)
        
        
        #for frame in self.timeframes:    
         #   add_atoms.set_ghost_coords(self, frame, new_atoms_info)
        add_atoms.set_all_ghost_coords(self,new_atoms_info,noise) 
        
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name)
    
        return 
    
    @staticmethod
    def set_ghost_connectivity(self,info):
        gc = dict()
        for j,v in info.items():
            gc[(v['bw'],j)] = (self.at_types[v['bw']],v['ty'])   
        self.ghost_connectivity = gc
        return
    
    @staticmethod
    def set_all_ghost_coords(self,info,noise=None):
        f,l,th,s,ir1,ir0,ir2 = add_atoms.serialize_info(info)
        self.unwrap_all()
        for frame in self.timeframes:      
            ghost_coords = np.empty((len(info),3),dtype=float)
            coords = self.get_coords(frame)
            add_atoms.set_ghost_coords_parallel(f,l,th,s,ir1,ir0,ir2,
                                                coords,ghost_coords)
            if noise is not None:
                noise_coords =  np.random.normal(0,noise,ghost_coords.shape) 
                ghost_coords += noise_coords
                logger.debug('Adding noise mean = {:4.3f}'.format(np.mean(np.abs(noise_coords))))
            self.timeframes[frame]['ghost_coords'] = ghost_coords
        return
    
    @staticmethod
    def serialize_info(info):
        n = len(info)
        f = np.empty(n,dtype=int)
        l = np.empty(n,dtype=float)
        th = np.empty(n,dtype=float)
        s = np.empty(n,dtype=int)
        ir1 = np.empty(n,dtype=int)
        ir0 = np.empty(n,dtype=int)
        ir2 = np.zeros(n,dtype=int)
        for j,(k,v) in enumerate(info.items()):
            f[j] = v['f']
            l[j] = v['l']
            th[j] = v['th']
            s[j] = v['s']
            ir1[j] = v['ir'][0]
            ir0[j] = v['ir'][1]
            if len(v['ir'])==3:
                ir2[j] = v['ir'][2]
        return f,l,th,s,ir1,ir0,ir2
    
    @staticmethod
    @jit(nopython=True,fastmath=True,parallel=True)
    def set_ghost_coords_parallel(f,l,th,s,ir1,ir0,ir2,
                                  coords,ghost_coords):
       
        N = f.shape[0]
        af = Analytical_Functions()
        for j in prange(N):
            cr1 = coords[ir1[j]]
            cr0 = coords[ir0[j]]
            cr2 = coords[ir2[j]]
            if   f[j] == 1:
                rn = af.position_hydrogen_analytically_cis(l[j],th[j],cr1,cr0,cr2,s[j])
            elif f[j] == 2:
                rn = af.position_hydrogen_analytically(l[j],th[j],cr1,cr0,cr2,s[j])
            elif f[j] == 3:
                rn = af.position_hydrogen_analytically_endgroup(l[j],th[j],cr1,cr0,s[j])
            ghost_coords[j] = rn
                 
        return

    @staticmethod
    def set_ghost_coords(self,frame,info):
       
        N = len(info)
        ghost_coords = np.empty((N,3),dtype=float)
        coords = self.get_whole_coords(frame)    
        
        for j,(k,v) in enumerate(info.items()):
            cr = coords[v['ir']]
            rn = v['func'](v['l'],v['th'],*cr,v['s'])
            ghost_coords[j] = rn
            
        self.timeframes[frame]['ghost_coords'] = ghost_coords
        return
    
    @staticmethod
    def assign_ghost_topol(self,info):
        n = len(info)
        
        gtypes = np.empty(n,dtype=object)
        gmol_names = np.empty(n,dtype=object)
        gat_ids = np.empty(n,dtype=int)
        gmol_ids = np.empty(n,dtype=int)
        #gatom_mass = np.empty(n,dtype=int)
        
        for i,(k,v) in enumerate(info.items()):
            gtypes[i] = v['ty']
            gmol_names[i] = v['res_nm']
            gmol_ids[i] = v['res']
            gat_ids[i] = k
            #gatom_mass[i] = v['mass']
             
            
        self.ghost_at_types = gtypes
        self.ghost_mol_names = gmol_names
        self.ghost_at_ids = gat_ids
        self.ghost_mol_ids = gmol_ids
        #self.ghost_atom_mass = gatom_mass
        
        return 
    
    @staticmethod
    def get_new_atoms_info(self,m,types):
        
        if m =='h':
            type_map = add_atoms.hydrogen_map
            self.mass_map.update({m+ty:elements_mass['H'] for ty in types})
        else:
            raise NotImplementedError('Currently the only option is to add hydrogens')
        
        bond_ids = np.array(list(self.connectivity.keys()))
        ang_ids = np.array(list(self.angles.keys()))
        
        at_types = self.at_types
        residues = self.mol_ids
        res_nms = self.mol_names
        
        
        jstart = self.at_types.shape[0]
        new_atoms_info = dict()
        
        fu = 'position_hydrogen_analytically'
        for t,ty in enumerate(at_types):
            if ty  not in types:
                continue
            tm = type_map[ty]
            i = self.at_ids[t]
            
            try:
                ir = ang_ids[np.where(ang_ids[:,1] == i)][0]
            except IndexError:
                try:
                    ir = bond_ids[np.where(bond_ids[:,0]==i)][0]
                    ir = np.array([ir[1],ir[0]])
                except IndexError:
                    try:
                        ir = bond_ids[np.where(bond_ids[:,1]==i)][0]
                    except IndexError as exc:
                        raise Exception('{}: Could not find bonds for atom {:d}, type = {:s}'.format(exc,i,ty))
                   
            finally:
                func = getattr(Analytical_Functions(),fu+tm[4])
            if tm[4] == '_cis': f = 1
            elif tm[4] =='' :  f = 2
            elif tm[4] =='_endgroup': f = 3
            for j in range(tm[0]):    
                new_atoms_info[jstart] = {'ir':ir,'l':tm[1],'th':tm[2]*np.pi/180.0,'s':tm[3][j], 'bw': i,
                                     'ty':m+ty,'res':residues[i],'res_nm':res_nms[i],
                                     'func':func,'f':f}
                jstart+=1
            
        return new_atoms_info

class Box_Additions():
    @staticmethod
    def zdir(box):
        return [box[2],0,-box[2]]
    @staticmethod
    def ydir(box):
        return [box[1],0,-box[1]]
    @staticmethod
    def xdir(box):
        return [box[0],0,-box[0]]
    def minimum_distance(box):
        zd = Box_Additions.zdir(box)
        yd = Box_Additions.ydir(box)
        xd = Box_Additions.xdir(box)
        ls = [np.array([x,y,z]) for x in xd for y in yd for z in zd]
        return ls
        
class Distance_Functions():
    @staticmethod
    def zdir(coords,zc):
        return np.abs(coords[:,2]-zc)
    @staticmethod
    def ydir(coords,yc):
        return np.abs(coords[:,1]-yc)
    @staticmethod
    def xdir(coords,xc):
        return np.abs(coords[:,0]-xc)
    
    @staticmethod
    def zdir__2side(coords,zc):
        return coords[:,2]-zc
    @staticmethod
    def ydir__2side(coords,yc):
        return coords[:,1]-yc
    @staticmethod
    def xdir__2side(coords,xc):
        return coords[:,0]-xc
    
    @staticmethod
    def distance(coords,c):
        d = np.zeros(coords.shape[0],dtype=float)
        distance_kernel(d,coords,c)
        return d
    
    @staticmethod
    def minimum_distance(coords1,coords2):
        d1 = np.empty(coords1.shape[0])
        d2 = np.empty(coords2.shape[0])
        smaller_distance_kernel(d1,d2,coords1,coords2)
        return d1

class bin_Volume_Functions():
    @staticmethod
    def zdir(box,bin_low,bin_up):
        binl = bin_up-bin_low
        return 2*box[0]*box[1]*binl
    
    @staticmethod
    def ydir(box,bin_low,bin_up):
        binl = bin_up-bin_low
        return 2*box[0]*box[2]*binl
    
    @staticmethod
    def xdir(box,bin_low,bin_up):
        binl = bin_up-bin_low
        return 2*box[1]*box[2]*binl
    
    @staticmethod
    def distance(box,bin_low,bin_up):
        dr = bin_up-bin_low
        rm = 0.5*(bin_up+bin_low)
        v = dr*(4*np.pi*rm**2)
        return  v
    
class Periodic_image_Functions():
    #functions return True for bridge
    @staticmethod
    def zdir(self,r0,re,dads):
        return abs(re[2]-r0[2]) > dads
    @staticmethod
    def ydir(self,r0,re,dads):
        return abs(re[1]-r0[1]) > dads
    @staticmethod
    def xdir(self,r0,re,dads):
        return abs(re[0]-r0[0]) > dads
    @staticmethod
    def distance(self,r0,re,dads): 
        raise NotImplementedError('This function is not yet implemented')
        return  ( abs(re) > box ).any() or ( abs(r0) > box ).any()

class Different_Particle_Functions():
    #functions return True for bridge
    @staticmethod
    def core_zyx_dir(d,r0,re,dads,CMs):
        x0=r0[d]
        xe =re[d]
        for i in range(CMs.shape[0]):
            for j in range(i+1,CMs.shape[0]):
                cmi = CMs[i][d] ; cmj = CMs[j][d]
                if abs(cmi-cmj)>dads: # if it is indeed a different particle
                    if (abs(cmi-x0)<dads and abs(cmj-xe)<dads) or \
                        (abs(cmi-xe)<dads and abs(cmj-x0)<dads):
                            return False
        return True
    
    @staticmethod
    def zdir(self,r0,re,dads,CMs):
        d = 2
        return Different_Particle_Functions.core_zyx_dir(d,r0,re,dads,CMs)
    @staticmethod
    def ydir(self,r0,re,dads,CMs):
        d = 1
        return Different_Particle_Functions.core_zyx_dir(d,r0,re,dads,CMs)
    @staticmethod
    def xdir(self,r0,re,dads,CMs):
        d = 0
        return Different_Particle_Functions.core_zyx_dir(d,r0,re,dads,CMs)
   


class Center_Functions():
    @staticmethod
    def zdir(c):
        return c[2]
    @staticmethod
    def ydir(c):
        return c[1]
    @staticmethod
    def xdir(c):
        return c[0]
    @staticmethod
    def distance(c):
        return c

class unit_vector_Functions():
    @staticmethod
    def zdir(r,c):
        uv = np.zeros((r.shape[0],3))
        uv[:,2] = 1
        return uv
    @staticmethod
    def ydir(r,c):
        uv = np.zeros((r.shape[0],3))
        uv[:,1] = 1
        return uv
    @staticmethod
    def xdir(r,c):
        uv = np.zeros((r.shape[0],3))
        uv[:,2] = 0
        return uv
    @staticmethod
    def distance(r,c):
        uv = r-c 
        return uv
    
class Analysis:
    
    def __init__(self,
                 trajectory_file, # gro for now
                 connectivity_info, #itp or mol2
                 gro_file = None,
                 memory_demanding=False,
                 types_from_itp=True):
        t0 = perf_counter()
        self.trajectory_file = trajectory_file
        self.connectivity_info = connectivity_info
        self.memory_demanding = memory_demanding
        self.types_from_itp = types_from_itp
        
        
        
        if '.gro' == trajectory_file[-4:]:
            self.gro_file = trajectory_file
            self.read_by_frame = self.read_gro_by_frame # function
            self.traj_opener = open
            self.traj_opener_args = (self.trajectory_file,'r')
        elif '.trr' == trajectory_file[-4:]:
            self.read_by_frame =  self.read_trr_by_frame # function
            self.traj_opener = GroTrrReader
            self.traj_opener_args = (self.trajectory_file,)
            if '.gro' == gro_file[-4:]: 
                self.gro_file = gro_file
            else:
                s = 'Your  trajectory file is {:s} while your gro file is {:s}.\n \
                Give the right format for the gro_file.\n \
                Only One frame is needed to get the \
                topology'.format(trajectory_file,gro_file)
                raise Exception(s)
            
        else:
            raise NotImplementedError('Trajectory file format ".{:s}" is not yet Implemented'.format(trajectory_file.split('.')[-1]))
        self.read_gro_topol()
        #self.get_masses()
        #find connectivity
        self.find_connectivity()
        self.analysis_initialization()
        
        self.timeframes = dict() # we will store the coordinates,box,step and time here
        tf = perf_counter()-t0
        #print_time(tf,inspect.currentframe().f_code.co_name)
        return 
    
    def analysis_initialization(self):
        
        t0 = perf_counter()
        #Now we want to find the connectivity,angles and dihedrals
        
        self.find_neibs()
        self.find_angles()
        self.find_dihedrals()
        #Find the ids (numpy compatible) of each type and store them
        self.keys_ids_per_type('connectivity')
        self.keys_ids_per_type('angles')
        self.keys_ids_per_type('dihedrals')
        
        self.find_unique_bond_types()
        self.find_unique_angle_types()
        self.find_unique_dihedral_types()
        self.dict_to_sorted_numpy('connectivity') #necessary to unwrap the coords efficiently
        
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name)
            
        return
    
    def update_connectivity(self,bid):
        bid,bt = self.sorted_id_and_type(bid)
        self.connectivity[bid] = bt
        return
    
    @staticmethod
    def get_timekey(time,t0):
        return round(time-t0,8)
    
    def dict_to_sorted_numpy(self,attr_name):
        t0 = perf_counter()
        
        attr = getattr(self,attr_name)
        if type(attr) is not dict:
            raise TypeError('This function is for working with dictionaries')
            
        keys = attr.keys()
        x = np.empty((len(keys),2),dtype=int)
        for i,k in enumerate(keys):
            x[i,0]=k[0] ; x[i,1]=k[-1]
            
        x = x[x[:,0].argsort()]
        setattr(self,'sorted_'+attr_name+'_keys',x-self.starts_from)
        
        tf = perf_counter() - t0 
        #print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return

    def keys_ids_per_type(self,attr_name):
        dictionary = getattr(self,attr_name)
        if type(dictionary) is not dict:
            raise TypeError('This function is for working with dictionaries')
        types = self.unique_values(dictionary.values())
        temp_ids = {i:[] for i in types}
        for k,v in dictionary.items():
            temp_ids[v].append(np.array(k))
        ids = {v : np.array(temp_ids[v])-self.starts_from for v in types}
        setattr(self,attr_name+'_ids_per_type',ids)
        return 
    
    def find_connectivity(self):
        t0 = perf_counter()
        conn = dict()
        if iterable(self.connectivity_info):
            for cf in self.connectivity_info:
                if '.itp' in cf:
                    self.read_atoms_and_connectivity_from_itp(cf)
                    
                else:   
                    raise NotImplementedError('Non itp files are not yet implemented')
                    '''
                    if len(cf) ==2:
    
                        a,t = self.sorted_id_and_type(cf)
                        conn[a] = t
                        explicit = True
                    else:
                        explicit = False
                        raise Exception(NotImplemented)
            try:
                if explicit:
                    self.connectivity = conn
                    return
            except NameError:
                pass
                    '''
                
        else:
            if '.itp' in self.connectivity_info:
                self.read_atoms_and_connectivity_from_itp(self.connectivity_info)
            else:
                raise NotImplementedError('Non itp files are not yet implemented')
     
        self.connectivity = dict()
        for j in np.unique(self.mol_ids):
            global_mol_at_ids = self.at_ids[self.mol_ids==j]
            res_nm = np.unique(self.mol_names[self.mol_ids==j])
            
            assert res_nm.shape ==(1,),'many names for a residue. Check code or topology file'
            
            res_nm = res_nm[0]
            local_connectivity = self.connectivity_per_resname[res_nm]
            
            for b in local_connectivity:
                
                id0 = self.loc_id_to_glob[j][b[0]]
                id1 = self.loc_id_to_glob[j][b[1]]
                conn_id,c_type = self.sorted_id_and_type((id0,id1))
                self.connectivity[conn_id] = c_type
        tf = perf_counter() - t0
        #print_time(tf,inspect.currentframe().f_code.co_name)
        return 

    def find_neibs(self):
        '''
        Computes first (bonded) neihbours of a system in dictionary format
        key: atom_id
        value: set of neihbours
        '''
        neibs = dict()
        for k in self.connectivity.keys(): 
            for i in k: neibs[i] = set() # initializing set of neibs
        for j in self.connectivity.keys():
            neibs[j[0]].add(j[1])
            neibs[j[1]].add(j[0])
        self.neibs = neibs
        return
    
    def box_mean(self):
        t0 = perf_counter()
        box = np.zeros(3)
        args = (box,)
        nframes = self.loop_trajectory('box_mean', args)
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return box/nframes
    
    def find_EndGroup_args(self,ty = None,serial=True):
        eargs = []
        if serial:
            for j,args in self.chain_args.items():
                 eargs.append(args[0]) ; eargs.append(args[-1])
            eargs = np.array(eargs)
        elif ty is not None:
            eargs = self.at_ids[self.at_types == ty]
        else:
            raise NotImplementedError('give either the type or serial=True if your atom chains are seriarly stored')
        self.EndGroup_args = eargs
        return
    
    def get_EndGroup_args(self):
        try:
            args = self.EndGroup_args
        except AttributeError as err:
            raise AttributeError('{}\nCall function "find_EndGroup_args" to set this attribute'.format(err))
        else:
            return args
        
    def box_variance(self):
        t0 = perf_counter()
        box_var = np.zeros(3)
        box_mean = self.box_mean()
        args = (box_var,box_mean**2)
        nframes = self.loop_trajectory('box_var', args)
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return box_var/nframes
    
    def frame_closer_tobox(self,target_box):
        mind = 10**9
        with self.traj_opener(*self.traj_opener_args) as ofile:
            t0 = perf_counter()
            nframes=0
            while(self.read_from_disk_or_mem(ofile, nframes)):
                box=self.timeframes[nframes]['boxsize']
                d = np.dot(target_box-box,target_box -box)**0.5
                if d < mind:
                    boxold = box
                    mind = d
                    frame_min = nframes
                if self.memory_demanding:
                    del self.timeframes[nframes]
                nframes+=1
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return frame_min
    
    def sorted_id_and_type(self,a_id):
        t = [self.at_types[i-self.starts_from] for i in a_id]
        if t[0]<=t[-1]:
            t = tuple(t)
        else:
            t= tuple(t[::-1])
        if a_id[0]<=a_id[-1]:
            a_id = tuple(a_id)
        else:
            a_id = tuple(a_id[::-1])
        return a_id,t
        
    def find_angles(self):
        '''
        Computes the angles of a system in dictionary format
        key: (atom_id1,atom_id2,atom_id3)
        value: object Angle
        Method:
            We search the neihbours of bonded atoms.
            If another atom is bonded to one of them an angle is formed
        We add in the angle the atoms that participate
        '''
        t0 = perf_counter()
        self.angles = dict()
        for k in self.connectivity.keys():
            #"left" side angles k[0]
            for neib in self.neibs[k[0]]:
                if neib in k: continue
                ang_id ,ang_type = self.sorted_id_and_type((neib,k[0],k[1]))
                if ang_id[::-1] not in self.angles.keys():
                    self.angles[ang_id] = ang_type
            #"right" side angles k[1]
            for neib in self.neibs[k[1]]:
                if neib in k: continue
                ang_id ,ang_type = self.sorted_id_and_type((k[0],k[1],neib))
                if ang_id[::-1] not in self.angles.keys():
                    self.angles[ang_id] = ang_type  
        tf = perf_counter()-t0
        #print_time(tf,inspect.currentframe().f_code.co_name)
        return

    def find_dihedrals(self):
        '''
        Computes dihedrals of a system based on angles in dictionary
        key: (atom_id1,atom_id2,atom_id3,atom_id4)
        value: object Dihedral
        Method:
            We search the neihbours of atoms at the edjes of Angles.
            If another atom is bonded to one of them a Dihedral is formed is formed
        We add in the angle the atoms that participate
        '''
        t0 = perf_counter()
        self.dihedrals=dict()
        for k in self.angles.keys():
            #"left" side dihedrals k[0]
            for neib in self.neibs[k[0]]:
                if neib in k: continue
                dih_id,dih_type = self.sorted_id_and_type((neib,k[0],k[1],k[2]))
                if dih_id[::-1] not in self.dihedrals:
                    self.dihedrals[dih_id] = dih_type
            #"right" side dihedrals k[2]
            for neib in self.neibs[k[2]]:
                if neib in k: continue
                dih_id,dih_type = self.sorted_id_and_type((k[0],k[1],k[2],neib))
                if dih_id[::-1] not in self.dihedrals:
                    self.dihedrals[dih_id] = dih_type
        tf = perf_counter()
        #print_time(tf,inspect.currentframe().f_code.co_name)
        return

    def correct_types_from_itp(self,itp_atoms):
        for i,t in enumerate(self.at_types.copy()):
            try:
                self.at_types[i] = itp_atoms[t]
            except KeyError:
                pass
        return

    def read_atoms_and_connectivity_from_itp(self,file):
        t0 = perf_counter()
        at_ids,at_types,res_num,res_name,atoms,cngr,charge,mass \
        = self.read_atoms_from_itp(file)
        
        t1 = perf_counter()
        if self.types_from_itp:
            self.correct_types_from_itp(atoms)
        if not hasattr(self, 'charge_map'):
            self.charge_map = charge
        else:
            for k,c in charge.items():
                self.charge_map[k] = c
        if not hasattr(self, 'mass_map'):
            self.mass_map = mass
        else:
            for k,m in mass.items():
                self.mass_map[k] = m
        bonds = self.read_connectivity_from_itp(file)
        
        connectivity_per_resname = {t:[] for t in np.unique(res_name) }
        
        for b in bonds:
            i0 = np.where(at_ids == b[0])[0][0]
            i1 = np.where(at_ids == b[1])[0][0]
            assert res_name[i0] == res_name[i1], 'Bond {:d} - {:d} is between two different residues'.format(i1,i2)
            
            res_nm = res_name[i0]
            connectivity_per_resname[res_nm].append(b)
            
        if not hasattr(self, 'connectivity_per_resname'):
            self.connectivity_per_resname = connectivity_per_resname
        else:
            for t,c in connectivity_per_resname.items():
                self.connectivity_per_resname[t] = c
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return 
    
    @staticmethod
    def read_atoms_from_itp(file):
        t0 = perf_counter()
        with open(file,'r') as f:
            lines = f.readlines()
            f.closed
        for j,line in enumerate(lines):
            if 'atoms' in line and '[' in line and ']' in line:
                jatoms = j
        at_ids = [] ; res_num=[];at_types = [] ;res_name = [] ; cngr =[] 
        charge = dict() ; mass=dict()
        connectivity = dict()
        atoms = dict()
        i=0
        for line in lines[jatoms+1:]:
            l = line.split()
            try:
                a = int(l[0])
            except:
                pass
            else:
                at_ids.append(i) #already int
                res_name.append(l[3])
                t = l[1]
                cngr.append(l[5])
                atoms[l[4]] = t
                at_types.append(t)
                charge[t] = float(l[6])
                mass[t] = float(l[7])
                res_num.append(int(l[2]))
                i+=1
            if '[' in line or ']' in line:
                break
            
        tf = perf_counter() - t0
        #print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return np.array(at_ids),np.array(at_types),np.array(res_num),np.array(res_name),atoms,cngr,charge,mass

    @staticmethod    
    def read_connectivity_from_itp(file):
        t0 = perf_counter()
        with open(file,'r') as f:
            lines = f.readlines()
            f.closed
            
        for j,line in enumerate(lines):
            
            if 'bonds' in line and '[' in line and ']' in line:
                jbonds = j
        bonds = []
        try:
            x= jbonds
        except UnboundLocalError:
            pass
        else:
            for line in lines[jbonds+1:]:
                l = line.split()
                try:
                    b = [int(l[0]),int(l[1])]
                except:
                    pass
                else:
                    bonds.append(b)
                if '[' in line or ']' in line:
                    break
        bonds = np.array(bonds)
        try:
            bonds-=bonds.min()
        except ValueError as e:
            logger.warning('Warning: File {:s} probably contains no bonds\n Excepted ValueError : {:}'.format(file,e))
        tf = perf_counter()
        return  bonds
            
    def read_gro_topol(self):
        with open(self.gro_file,'r') as f:
            l = f.readline()
            natoms = int(f.readline().strip())
            #allocate memory
            mol_ids = np.empty(natoms,dtype=int)
            mol_nms = np.empty(natoms,dtype=object)
            at_tys  = np.empty(natoms,dtype=object)
            at_ids  = np.empty(natoms,dtype=int)
            for i in range(natoms):
                line = f.readline()
                mol_ids[i] = int(line[0:5].strip())
                mol_nms[i] = line[5:10].strip()
                at_tys[i] = line[10:15].strip()
                #at_ids[i] = int(line[15:20].strip())
                at_ids[i] = i
            f.close()
        
        starts_from = at_ids.min()
        
        loc_id_to_glob = dict() ; glob_id_to_loc = dict()
        for j in np.unique(mol_ids):
            loc_id_to_glob[j] = dict()
            glob_id_to_loc[j] = dict()
            filt = mol_ids== j
            res_nm = np.unique(mol_nms[filt])
            if res_nm.shape !=(1,):
                raise ValueError('many names for a residue, res_id = {:d}'.format(j))
            else:
                res_nm = res_nm[0]
            g_at_id = at_ids[filt]
            
            for i,g in enumerate(g_at_id):
                loc_id = i+starts_from
                loc_id_to_glob[j][loc_id] = g
                glob_id_to_loc[j][g] = loc_id
        
        self.starts_from = starts_from
        self.loc_id_to_glob = loc_id_to_glob
        self.glob_id_to_loc = glob_id_to_loc
        
        
        self.mol_ids = mol_ids
        self.mol_names = mol_nms
        self.at_types = at_tys
        self.at_ids = at_ids
        return 
    
    def charge_from_maps(self):
        self.charge_map.update(maps.charge_map)
        return
    
    @property
    def natoms(self):
        return self.at_ids.shape[0]
    
    @staticmethod
    def unique_values(iterable):
        try:
            iter(iterable)
        except:
            raise Exception('Give an iterable variable')
        else:
            un = []
            for x in iterable:
                if x not in un:
                    un.append(x)
            return un

    def find_unique_bond_types(self):
        self.bond_types = self.unique_values(self.connectivity.values())
    def find_unique_angle_types(self):
        self.angle_types = self.unique_values(self.angles.values())
    def find_unique_dihedral_types(self):
        self.dihedral_types = self.unique_values(self.dihedrals.values())
    
    def get_masses(self):
        t0 = perf_counter()
        mass = np.empty(self.natoms,dtype=float)
        for i in range(self.natoms):
            mass[i] = self.mass_map[self.at_types[i]]
        self.atom_mass = mass
        tf = perf_counter() -t0
        #print_time(tf,inspect.currentframe().f_code.co_name)
        return
    
    @staticmethod
    def get_equal_from_string(s,v,make=float):
        try:
            x = s.split(v+'=')[-1].split()[0]
        except:
            x = s.split(v+' =')[-1].split()[0]
        return make(x) 
    
    def read_gro_by_frame(self,ofile,frame):
        line = ofile.readline()
        l = line.strip().split()
        if len(l)==0:
            return False
        #first line
        try:
            time = self.get_equal_from_string(line.strip(),'t')
        except:
            logger.warning('Warning: in gro file. There is no time info')
            time = 0
        try:
            step = self.get_equal_from_string(line.strip(),'step',int)
        except:
            step = 0
            logger.warning('Warning: in gro file. There is no step info')
        self.timeframes[frame] = {'time':time,'step':step}
        # second line
        natoms = int(ofile.readline().strip())
        if natoms != self.natoms:
            raise ValueError('This frame has {} atoms instead of {}'.format(natoms,self.natoms))
        
        #file 
        coords = np.empty((natoms,3),dtype=float)
        for i in range(natoms):
            line = ofile.readline()
            l=line[20:44].split() 
            coords[i,0] = float(l[0]) 
            coords[i,1] = float(l[1])  
            coords[i,2] = float(l[2])
        
        box = np.array(ofile.readline().strip().split(),dtype=float)
        
        self.timeframes[frame]['coords'] = coords
        self.timeframes[frame]['boxsize'] = box
        return True
    
    def read_trr_by_frame(self,ofile,frame):
        try:
            header,data = ofile.read_frame()
        except EOFError as e:
            raise e
        self.timeframes[frame] = header
        self.timeframes[frame]['boxsize'] = np.diag(data['box'])
        self.timeframes[frame]['coords'] = data['x']
        return True
    
    def read_from_disk_or_mem(self,ofile,frame):
        if self.memory_demanding:
            try:
                return self.read_by_frame(ofile, frame)
            except EOFError:
                return False
        elif frame in self.timeframes.keys():
            self.is_the_frame_read =True
            return True
        else:
            try:
                if self.is_the_frame_read:
                    return False
            except AttributeError:
                return self.read_by_frame(ofile, frame)
            
    def read_trr_file(self):
        t0 = perf_counter()
        with GroTrrReader(self.trajectory_file) as ofile:
            end = False
            nframes = 0
            while( end == False):
                try:
                    self.read_trr_by_frame(ofile, nframes)
                except EOFError:
                    end = True
                else:
                    if self.memory_demanding:
                        del self.timeframes[nframes]
                    nframes += 1
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return 

    def read_gro_file(self):
        t0 = perf_counter()
        with open(self.trajectory_file,'r') as ofile:
            nframes =0
            while(self.read_gro_by_frame(ofile,nframes)):
                
                if self.memory_demanding:
                    del self.timeframes[nframes]
                nframes+=1
            ofile.close()
            tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return 
    
    def read_file(self):    
        if   'gro' in self.trajectory_file: self.read_gro_file()
        elif 'trr' in self.trajectory_file: self.read_trr_file()
        return 
    
    def write_gro_file(self,fname=None,option='',frames=None):
        t0 = perf_counter()
        options = ['','unwrap','transmiddle']
        if option not in options:
            raise ValueError('Available options are : {:s}'.format(', '.join(options)))
        
        if fname is None:
            fname = 'Analyisis_written__'+''.join(self.trajectory_file.split('.')[0:-1])+'.gro'
        with open(fname,'w') as ofile:
            for frame,d in self.timeframes.items():
                if frames is not None:
                    if  frame <frames[0] or frame>frames[1]:
                        continue
                if option == 'unwrap':
                    coords = self.get_whole_coords(frame)
                elif option =='transmiddle':
                    coords = self.translate_particle_in_box_middle(self.get_coords(frame),
                                                                   self.get_box(frame))
                elif option=='':
                
                    coords = d['coords']
                else:
                    raise NotImplementedError('option "{:}" Not implemented'.format(option))
                self.write_gro_by_frame(ofile,
                                        coords, d['boxsize'],
                                        time = d['time'], 
                                        step =d['step'])
            ofile.close()
        tf = perf_counter() -t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return

    def write_gro_by_frame(self,ofile,coords,box,name='gro_by_frame',time=0,step=0):
        ofile.write('{:s},  t=   {:4.3f}  step=   {:8.0f} \n'.format(name,time,step))
        ofile.write('{:6d}\n'.format(coords.shape[0]))
        for i in range(coords.shape[0]):
            c = coords[i]
            ofile.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n'\
            % (self.mol_ids[i],self.mol_names[i], self.at_types[i],self.at_ids[i%100000] ,c[0],c[1] ,c[2] ))
        ofile.write('%f  %f  %f\n' % (box[0],box[1],box[2]))
        return  
    
    def write_residues(self,res,fname='selected_residues.gro',
                       frames=(0,0),box=None,boxoff=0.4):
        with open(fname,'w') as ofile:
            fres = np.isin(self.mol_ids, res)
            
            for frame in self.timeframes:
                if frames[0] <= frame <= frames[1]:

                    coords = self.get_whole_coords(frame) [fres]
                    at_ids = self.at_ids[fres] 
                    ofile.write('Made by write_residues\n')
                    ofile.write('{:6d}\n'.format(coords.shape[0]))
                    for j in range(coords.shape[0]):
                        i = at_ids[j]
                        c = coords[j]
                        ofile.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n'\
                        % (self.mol_ids[i],self.mol_names[i], self.at_types[i]
                           ,self.at_ids[i%100000] ,c[0],c[1] ,c[2] ))
                    if box is None: 
                        box = coords.max(axis=1) - coords.min(axis=1) + boxoff
                    ofile.write('%f  %f  %f\n' % (box[0],box[1],box[2]))  
            
            ofile.closed
        return
    def write_residue(self,res,frames=(0,0),boxoff=0.4):
        with open('res{}.gro'.format(res),'w') as ofile:
            fres = self.mol_ids == res
            
            for frame in self.timeframes:
                if frames[0] <= frame <= frames[1]:

                    coords = self.get_whole_coords(frame) [fres]
                    at_ids = self.at_ids[fres] 
                    ofile.write('Residue {:d}\n'.format(res))
                    ofile.write('{:6d}\n'.format(coords.shape[0]))
                    for j in range(coords.shape[0]):
                        i = at_ids[j]
                        c = coords[j]
                        ofile.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n'\
                        % (self.mol_ids[i],self.mol_names[i], self.at_types[i]
                           ,self.at_ids[i%100000] ,c[0],c[1] ,c[2] ))
                    
                    box = coords.max(axis=1) - coords.min(axis=1) + boxoff
                    ofile.write('%f  %f  %f\n' % (box[0],box[1],box[2]))  
            
            ofile.closed
        return
            
    def unwrap_coords(self,coords,box):   

        unc = coords.copy()
        b2 =  box/2
        k0 = self.sorted_connectivity_keys[:,0]
        k1 = self.sorted_connectivity_keys[:,1]
        n = np.arange(0,k0.shape[0],1,dtype=int)
        dim = np.array([0,1,2],dtype=int)

        unc = unwrap_coords_kernel(unc, k0, k1, b2, n, dim, box)
        
        return unc

    def unwrap_all(self):
        for k in self.timeframes:
            coords = self.timeframes[k]['coords']
            box =self.timeframes[k]['boxsize']
            self.timeframes[k]['coords'] = self.unwrap_coords(coords, box)
        return
    

    def loop_trajectory(self,fun,args):
        funtocall = getattr(coreFunctions,fun)
        if len(self.timeframes) == 0:# or self.memory_demanding:
            with self.traj_opener(*self.traj_opener_args) as ofile:
                nframes=0
                while(self.read_from_disk_or_mem(ofile,nframes)):
                    funtocall(self,nframes,*args)      
                    if self.memory_demanding:
                        del self.timeframes[nframes]
                    nframes+=1
        else:
            nframes = self.loop_timeframes(funtocall,args)
        return nframes
    @property
    def first_frame(self):
        return list(self.timeframes.keys())[0]
    
    def cut_timeframes(self,num_start=None,num_end=None):
        if num_start is None and num_end is None:
            raise Exception('Give either a number to cut from the start or from the end for the timeframes dictionary')
        if num_start is not None:
            i1 = num_start
        else:
            i1 =0
        if num_end is not None:
            i2 = num_end
        else:
            i2 = len(self.timeframes)
        new_dict = self.dict_slice(self.timeframes,i1,i2)
        if len(new_dict) ==0:
            raise Exception('Oh dear you have cut all your timeframes from memory')
            
        self.timeframes = new_dict
        return 
    
    @staticmethod
    def dict_slice(d,i1,i2):
        return {k:v for i,(k,v) in enumerate(d.items()) if i1<=i<i2 }
   
    def loop_timeframes(self,funtocall,args):
        for frame in self.timeframes:
            funtocall(self,frame,*args)
        nframes = len(self.timeframes)
        return nframes
    


    
    def calc_pair_distribution(self,binl,dmax,type1=None,type2=None,
                               density='number'):
        
        t0 = perf_counter()   
        
        if density not in ['number','probability']:
            raise NotImplementedError('density = {:s} is not Impmemented'.format(density))
            
        bins = np.arange(0,dmax+binl,binl)
        gofr = np.zeros(bins.shape[0]-1,dtype=float)
        if type1 is None and type2 is None:
            func = 'gofr'
            args = ()
            nc1 = self.get_coords(0).shape[0]
            nc2 = (nc1-1)/2
        elif type1 is not None and type2 is None:
            func = 'gofr_type_to_all'
            fty1 = type1 == self.at_types
            nc1 = self.get_coords(0)[fty1].shape[0]
            nc2 = self.get_coords(0).shape[0]
            args = (fty1,)
        elif type1 is None and type2 is not None:
            func ='gofr_type_to_all'
            fty2 = type2 == self.at_types
            nc1 = self.get_coords(0).shape[0]
            nc2 = self.get_coords(0)[fty2].shape[0]
            args =(fty2,)
        elif type1==type2:
            func= 'gofr_type'
            fty = type1 == self.at_types
            nc1 =  self.get_coords(0)[fty].shape[0]
            nc2 =(nc1-1)/2
            args = (fty,)
        else:
            func = 'gofr_type_to_type'
            fty1,fty2 = type1 == self.at_types, type2 == self.at_types
            nc1 = self.get_coords(0)[fty1].shape[0]
            nc2 = self.get_coords(0)[fty2].shape[0]
            args= (type1 == self.at_types, type2 == self.at_types)
        
        args = (*args,bins,gofr)
        
        nframes = self.loop_trajectory(func, args)


        gofr/=nframes*nc2
        cb = center_of_bins(bins)
        if density=='number':
            vshell = 4*np.pi*(cb[1]-cb[0])*cb**2
            gofr/=vshell
        elif density =='probability':
            gofr/=nc1
        
        pair_distribution = {'d':cb, 'gr':gofr }
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return pair_distribution
    
    
    def get_coords(self,frame):
        return self.timeframes[frame]['coords']
    
    def get_box(self,frame):
        return self.timeframes[frame]['boxsize']
    
    def get_time(self,frame):
        return self.timeframes[frame]['time']
    
    def frame_coords(self,frame):
        coords = self.get_coords(frame)
        return coords
   
    def get_whole_coords(self,frame):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.unwrap_coords(coords, box)
        return coords
    
    def bond_distance_matrix(self,ids):
        t0 = perf_counter()
        size = ids.shape[0]
        distmatrix = np.zeros((size,size),dtype=int)
        for j1,i1 in enumerate(ids):
            nbonds = self.bond_distance_id_to_ids(i1,ids)
            distmatrix[j1,:] = nbonds
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return distmatrix
    def bond_distance_id_to_ids(self,i,ids):
        chunk = {i}
        n = ids.shape[0]
        nbonds = np.ones(n)*(-1)
        incr_bonds = 0
        new_neibs = np.array(list(chunk))
        while new_neibs.shape[0]!=0:
            f = np.zeros(ids.shape[0],dtype=bool)
            numba_isin(ids,new_neibs,f)
            nbonds[f] = incr_bonds
            new_set = set()
            for ii in new_neibs:
                for neib in self.neibs[ii]:
                    if neib not in chunk:
                        new_set.add(neib)
                        chunk.add(neib)
            new_neibs = np.array(list(new_set))
            incr_bonds+=1
        return nbonds

        
                
    def ids_nbondsFrom_args(self,ids,args):
        
        n = ids.shape[0]
        nbonds = np.ones(n)*10**10
        
        #old_chunk = set() ;
        new_neibs = args.copy()
        chunk = set(args)
        incr_bonds=0
        #same_set = False
        fnotin_args = np.logical_not(np.isin(ids,args))

        while  new_neibs.shape[0] !=0:
            
            #f = np.logical_and(np.isin(ids,new_neibs),fnotin_args)
            f = np.zeros(ids.shape[0],dtype=bool)
            numba_isin(ids,new_neibs,f)
            f = np.logical_and(f,fnotin_args)
            
            
            nbonds [ f ] = incr_bonds
            new_set = set()
            for ii in new_neibs:
                for neib in self.neibs[ii]:                    
                    if neib in chunk: continue        
                    new_set.add(neib)
                    chunk.add(neib)
            
            new_neibs = np.array(list(new_set)) 
            incr_bonds += 1
            
        return nbonds
    
class Analysis_Confined(Analysis):
    
    
    def __init__(self, trajectory_file,   
                 connectivity_info,       # itp or mol2
                 conftype,
                 gro_file = None,
                 memory_demanding=True, 
                 particle_filt=None,
                 particle_name=None,
                 pol_filt=None,
                 pol_name=None):
        super().__init__(trajectory_file,
                         connectivity_info,
                         gro_file,
                         memory_demanding)
        self.conftype = conftype
        self.particle_name = particle_name
        self.particle_filt = particle_filt
        self.pol_name = pol_name
        self.pol_filt = pol_filt
        
        self.confined_system_initialization()
        

        return
    
    ############## General Supportive functions Section #####################
    def find_args_per_residue(self,filt,attr_name):
        args = dict()
        for j in np.unique(self.mol_ids[filt]):
            x = np.where(self.mol_ids==j)[0]
            args[j] = x
        setattr(self,attr_name,args)
        setattr(self,'N'+attr_name, len(args))
        return 
    
    def find_connectivity_per_chain(self):
        cargs =dict()
        x = self.sorted_connectivity_keys
        for j,args in self.chain_args.items():
            f1 = np.isin(x[:,0],args)
            f2 = np.isin(x[:,1],args)
            f = np.logical_or(f1,f2)
            cargs[j] = x[f]
        self.connectivity_per_chain = cargs
        return
    
    def find_particle_filt(self):
        
        if self.particle_name is not None:
            self.particle_filt = self.mol_names == self.particle_name # it gets a filter form
       
        else:
            raise Exception('Give particle_name or particle_filt explicitly')
        logger.info('Number of particle atoms: {:5d}'.format(np.count_nonzero(self.particle_filt)))
        return 
    
    def find_pol_filt(self):
        
        if self.pol_name is not None:
            self.pol_filt = self.mol_names == self.pol_name # it gets a filter form
        else:
            raise Exception('Give mol_name or mol_ids explicitly')
        logger.info('Number of adsorbent atoms: {:5d}'.format(np.count_nonzero(self.pol_filt)))
        return
       
    def translate_particle_in_box_middle(self,coords,box):
        particle_cm = CM( coords[self.particle_filt], self.atom_mass[self.particle_filt])
        coords += box/2 - particle_cm
        coords  = implement_pbc(coords,box)
        return coords
    
    def frame_coords(self,frame):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.translate_particle_in_box_middle(coords, box)
        return coords
   
    def get_whole_coords(self,frame):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.translate_particle_in_box_middle(coords, box)
        coords = self.unwrap_coords(coords, box)
        return coords
   
    def confined_system_initialization(self):
        
        t0 = perf_counter()
        self.find_particle_filt()
        self.nparticle = self.mol_ids[self.particle_filt].shape[0]
    
        self.find_pol_filt()
        self.npol = self.mol_ids[self.pol_filt].shape[0]
    
        self.get_masses()
        self.unique_atom_types = np.unique(self.at_types)
        
        self.dfun = self.get_class_function(Distance_Functions,self.conftype)
        self.box_add = self.get_class_function(Box_Additions,self.conftype)
        self.volfun = self.get_class_function(bin_Volume_Functions,self.conftype)
        self.centerfun = self.get_class_function(Center_Functions,self.conftype)
        self.is_periodic_image = self.get_class_function(Periodic_image_Functions,self.conftype)
        self.is_different_particle = self.get_class_function(Different_Particle_Functions,self.conftype)
        self.unit_vectorFun = self.get_class_function(unit_vector_Functions,self.conftype)
        
        self.find_args_per_residue(self.pol_filt,'chain_args')
        self.find_connectivity_per_chain()
        self.find_args_per_residue(self.particle_filt,'particle_args')
        self.nparticles = len(self.particle_args.keys())
        
        self.all_args = np.arange(0,self.natoms,1,dtype=int)
        
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return
       
    def get_class_function(self,_class,fun,inplace=False):
        fun = getattr(_class,fun)
        if inplace:
            attr_name = _class+'_function'
            setattr(self,attr_name, fun)
        return fun
     
    def get_frame_essentials(self,frame):
        coords = self.frame_coords(frame)
        box  = self.get_box(frame)          
        d,cs = self.get_distFromParticle(coords)
        return coords,box,d,cs
    
    def get_unwrappedframe_essentials(self,frame):
        coords = self.get_whole_coords(frame)
        box  = self.get_box(frame)          
        d,cs = self.get_distFromParticle(coords)
        return coords,box,d,cs


    
    def get_PoldistFromParticle(self,coords):
        cs = self.centerfun(CM( coords[self.particle_filt], 
                               self.atom_mass[self.particle_filt]))
        d = self.dfun(coords[self.pol_filt],cs)
        return d,cs
    def get_distFromParticle(self,coords):
        cs = self.centerfun(CM( coords[self.particle_filt], 
                               self.atom_mass[self.particle_filt]))
        d = self.dfun(coords,cs)
        return d,cs
    def get_particle_cm(self,coords):
        cm =self.centerfun(CM( coords[self.particle_filt], 
                               self.atom_mass[self.particle_filt]))
        return cm
    @staticmethod
    def get_layers(dads,dmax,binl):
        bins = np.arange(dads,dmax+binl,binl)
        dlayers = [(0,dads)]
        nbins = len(bins)-1
        for i in range(0,nbins):
            dlayers.append((bins[i],bins[i+1]))
        return dlayers

    def ids_from_topology(self,topol_vector):
        inter = len(topol_vector)
        if inter == 2: 
            ids = self.connectivity_ids_per_type
        elif inter == 3: 
            ids = self.angles_ids_per_type
        elif inter == 4: 
            ids = self.dihedrals_ids_per_type
        else:
            raise Exception('Large topology vectors with size >= {} are not Implemented'.format(inter))
        topol_vector = tuple(topol_vector)
      
        if topol_vector in ids: tp = topol_vector
        elif topol_vector[::-1] in ids: tp = topol_vector[::-1]
        else:
            raise Exception('{} not in {}'.format(topol_vector,list(ids)))
        
        arr0 = ids[tp][:,0] ; arr1 = ids[tp][:,-1]
        
        return arr0,arr1
   
    def ids_from_keyword(self,keyword,exclude=[]):
        if keyword in ['4',4,'1-4']:
            ids = self.dihedrals_ids_per_type
        if keyword in ['3',3,'1-3']:
            ids = self.angles_ids_per_type
        if keyword in ['2',2,'1-2']:
            ids = self.connectivity_ids_per_type
        ids1 = np.empty(0,dtype=int)
        ids2 = np.empty(0,dtype=int)
        for k,i in ids.items():
            if k in exclude:
                continue
            ids1 = np.concatenate( (ids1,i[:,0]) )
            ids2 = np.concatenate( (ids2,i[:,-1]) )
        return ids1,ids2
    
    def find_vector_ids(self,topol_vector,exclude=[]):
        #t0 = perf_counter()
        ty = type(topol_vector)
        if ty is list or ty is tuple:
            ids1,ids2 = self.ids_from_topology(topol_vector)
        if ty is str or ty is int:
            ids1,ids2 = self.ids_from_keyword(topol_vector,exclude)

        #logger.info('time to find vector list --> {:.3e}'.format(perf_counter()-t0))
        return ids1,ids2
    
    ###############End of General Supportive functions Section#########

    ############### Conformation Calculation Supportive Functions #####
   
    def check_if_ends_belong_to_periodic_image(self,istart,iend,periodic_image_args):
        
        
        #perimage = self.is_periodic_image(self,r0,re,dads)
        
        e = iend in periodic_image_args 
        s = istart in periodic_image_args
        
        return (e and not s) or (s and not e)
    
    def check_if_ends_belong_to_different_particle(self,coords,istart,iend,dads):
        #logger.warning('WARNING Function {:s}: This Function was never examined in test cases'.format(inspect.currentframe().f_code.co_name))
        r0 = coords[istart]
        re = coords[iend]
        CMs = np.empty((self.nparticles,3),dtype=float)
        for i,(k,args) in enumerate(self.particle_args.items()):
            CMs[i] = CM(coords[args],self.atom_mass[args])
            
        return self.is_different_particle(self,r0,re,dads,CMs)
   
    def is_bridge(self,coords,istart,iend,dads,periodic_image_args):
        
        if self.nparticles !=1:
            same_particle = self.check_if_ends_belong_to_different_particle(coords, istart, iend,dads,)
        else:
            same_particle = True
            
        if same_particle:  
            #logger.debug('istart = {:d}, iend = {:d}'.format(istart,iend))
            return self.check_if_ends_belong_to_periodic_image( istart, iend, periodic_image_args)
        else:
            #logger.info('istart = {:d} , iend = {:d}  Belong to differrent particle'.format(istart,iend))
            return True
        return False
    
    def get_filt_train(self,dads,coords,box):
        ftrain = False
       
        for L in self.box_add(box):
            cm = self.get_particle_cm(coords+L)
            d = self.dfun(coords,cm)
            ftrain = np.logical_or(ftrain,np.less_equal(d,dads))
        ftrain = np.logical_and(ftrain,self.pol_filt)
        
        self_trains = np.less_equal(
            self.dfun(coords,self.get_particle_cm(coords)),dads)
        image_trains = np.logical_and(ftrain,np.logical_not(self_trains))
        
        return ftrain,image_trains
    
    def get_minimum_distance_from_particle(self,coords,box):
        d = np.ones(coords.shape[0])*float('inf')
        for L in self.box_add(box):
            cm = self.get_particle_cm(coords+L)
            d = np.minimum(d,self.dfun(coords,cm))
        return d
    
    def conformations(self,dads,coords,box):
        
        ftrain,image_trains = self.get_filt_train(dads, coords, box)
        args_train = np.nonzero(ftrain)[0]
        periodic_image_args = set(np.nonzero(image_trains)[0])
        #logger.debug('Number of periodic image trains ={:d}\n Number of trains = {:d}'.format(len(periodic_image_args),args_train.shape[0]))
        #ads_chains
        ads_chains = np.unique(self.mol_ids[ftrain])
        #check_occurances(ads_chains)
        fads_chains = np.isin(self.mol_ids,ads_chains)
        args_ads_chain_atoms = np.nonzero(fads_chains)[0]
        
        #tail_loop_bridge
        f_looptailbridge = np.logical_and(fads_chains, np.logical_not(ftrain))
        args_looptailbridge = np.nonzero(f_looptailbridge)[0]
        
        args_tail = np.empty(0,dtype=int) ; 
        args_bridge = np.empty(0,dtype=int) ; 
        args_loop  = np.empty(0,dtype=int)
        
        for j in ads_chains:
            args_chain = self.chain_args[j]
            connectivity_args = self.connectivity_per_chain[j]
            nch = args_chain.shape[0]
            
            #1) Identiyfy connectivity nodes
            nodes = []
            for c in connectivity_args:
                ft = ftrain[c]
                if ft[0] and not ft[1]: nodes.append( (c[1],c[0]) )
                if not ft[0] and ft[1]: nodes.append( (c[0],c[1]) )
            
            #2) Loop over nodes and identify chunks
            for node in nodes:

                chunk = {node[0]}
                istart = node[1]
                
                old_chunk = set() ; loopBridge = False ; found_iend = False
                new_neibs = chunk.copy()
                
                while old_chunk != chunk:
                    old_chunk = chunk.copy()
                    new_set = set()
                    for ii in new_neibs:
                        for neib in self.neibs[ii]:
                            
                            if neib == istart: continue
    
                            if ftrain[neib]: 
                                loopBridge = True
                                
                                if not found_iend:
                                    iend = neib 
                                    found_iend = True 
                                continue
                             
                            if neib not in chunk:        
                                new_set.add(neib)
                                chunk.add(neib)
                    new_neibs = new_set
    
                if not found_iend:
                    try: del iend
                    except: pass
  
                
                chunk = np.array(list(chunk),dtype=int)
                assert not ftrain[chunk].any(), \
                    'chunk in train chain {:d}, node {}, chunk size = {:d}\
                    ,istart = {:d}, iend ={:d} \n\n chunk =\n {} \n\n'.format(j,
                    node,chunk.shape[0],istart,iend,chunk)
                assert fads_chains[chunk].all(),\
                    'chunk out of adsorbed chains, j = {:d} ,\
                        node = {} n\n chunk \n {} \n\n'.format(j,
                        node,chunk)
    
                if loopBridge:
                    if not self.is_bridge(coords,istart,iend,dads,periodic_image_args):    
                        args_loop = np.concatenate( (args_loop, chunk) )             
                    else:
                        logger.debug('chain = {:d}, chunk | (istart,iend) = ({:d}-{:d}) is bridge'.format(j,istart,iend))
                        args_bridge = np.concatenate( (args_bridge, chunk) )
                else:
                    args_tail = np.concatenate( (args_tail, chunk) )

        #args_tail = np.unique(args_tail)
        args_loop = np.unique(args_loop)
        args_bridge = np.unique(args_bridge)
        assert not np.isin(args_train,args_tail).any(),\
        'tails in trains, there are {:d}'.format(np.count_nonzero(np.isin(args_train,args_tail)))
        
        assert not np.isin(args_train,args_loop).any(),\
        'loops in trains, there are {:d}'.format(np.count_nonzero(np.isin(args_train,args_loop)))
        
        assert not np.isin(args_tail,args_loop).any(),\
        'loops in tails, there are {:d}'.format(np.count_nonzero(np.isin(args_tail,args_loop)))
        
        
        assert not np.isin(args_train,args_bridge).any(),\
        'bridge in trains, there are {:d}'.format(np.count_nonzero(np.isin(args_train,args_bridge)))
        
        assert not np.isin(args_tail,args_bridge).any(),\
        'bridge in tails, there are {:d}'.format(np.count_nonzero(np.isin(args_tail,args_bridge)))
        
        assert args_train.shape[0] + args_tail.shape[0] +\
               args_loop.shape[0] + args_bridge.shape[0] \
               == np.count_nonzero(fads_chains) , 'Confomormations do not sum up correnctly,\
                   correct sum = {:d}, sum = {:d}'.format( args_ads_chain_atoms.shape[0],
                   args_train.shape[0] + args_tail.shape[0] +\
                   args_loop.shape[0] + args_bridge.shape[0] 
                                                      )
    
        
        return ads_chains,args_train,args_tail,args_loop,args_bridge
    
    def conf_chunks(self,args):
        #t0 = perf_counter()
        set_args = set(args)
        chunks = []
        aold = -1
        
        while(len(set_args)>0):

            a = set_args.pop()
            set_args.add(a)
            #assert a != aold,'while loop stuck on argument = {}'.format(a)
            old_set_a = set()
            new_set_a = {a}
                    
            new_neibs = new_set_a.copy()
            
            while new_set_a != old_set_a:
                old_set_a = new_set_a.copy()
                
                for j in new_neibs.copy():
                    new_neibs = set()
                    for neib in self.neibs[j]:
                        if neib in set_args:
                            new_set_a.add(neib)
                            if neib not in old_set_a:
                                new_neibs.add(neib)
                
                

            chunks.append(new_set_a)
            set_args.difference_update(new_set_a)
        
            #aold = a
        #print_time(perf_counter()-t0,'conf_chunks')
        return chunks
    ############### End of Conformation Calculation Supportive Functions #####
    
    ######## Main calculation Functions for structural properties ############      
    
    def calc_density_profile(self,binl,dmax,mode='mass',option='',flux=None):
        
        t0 = perf_counter()
        
        #initialize
        ##############
        scale = 1.660539e-3 if mode == 'mass' else 1.0
        density_profile = dict()
        
        if dmax is None:
            NotImplemented
        bins  =   np.arange(0, dmax+binl, binl)
        nbins = len(bins)-1
        rho = np.zeros(nbins,dtype=float)
        mass_pol = self.atom_mass[self.pol_filt] 
        
        if option == '__pertype':
            rho_per_atom_type = {t:np.zeros(nbins,dtype=float) for t in self.unique_atom_types }
            ftt = {t: t == self.at_types[self.pol_filt] for t in rho_per_atom_type.keys() }
            args = (nbins,bins,rho,rho_per_atom_type,ftt)
        elif option =='':
            args = (nbins,bins,rho)
        elif option =='__2side':
            rho_down = np.zeros(nbins,dtype=float)
            args =(nbins,bins,rho,rho_down)
        
        if mode =='mass': args =(*args,mass_pol)
        
        if flux is not None and flux !=False:
            density_profile.update(self.calc_density_profile(binl,dmax,mode,option))
            rho_mean = density_profile['rho'].copy()
            rho_mean/=scale
            args = (nbins,bins,rho,mass_pol,rho_mean**2)
            func =  mode+'_density_profile'+'_flux'
        else:
            func =  mode+'_density_profile'+option    
        
        #############
        
        #calculate
        #############
        nframes = self.loop_trajectory(func, args)
        #############
        
        #post_process
        ##############
        
        if flux is not None and flux !=False:
            density_profile['rho_flux'] = rho*scale**2/nframes
        else:
            rho*=scale/nframes 
            d_center = center_of_bins(bins)
            density_profile.update({'d':d_center})
            
            density_profile.update({'rho':rho})
            if option =='__pertype':
                for t,rhot in rho_per_atom_type.items(): 
                    density_profile[t] = rhot*scale/nframes 
            elif option =='__2side':
                rho_down *=scale/nframes
                density_profile.update({'rho_up':rho,'rho_down':rho_down,'rho':0.5*(rho+rho_down)})
        
        #############
        
        tf = perf_counter() -t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return density_profile
    

    
    def calc_P2(self,binl,dmax,topol_vector):
        t0 = perf_counter()

        bins  =   np.arange(0, dmax, binl)
        dlayers=[]
        for i in range(0,len(bins)-1):
            dlayers.append((bins[i],bins[i+1]))
        d_center = [0.5*(b[0]+b[1]) for b in dlayers]
        
        ids1, ids2 = self.find_vector_ids(topol_vector)
        nvectors = ids1.shape[0]
        logger.info('topol {}: {:d} vectors  '.format(topol_vector,nvectors))
        nlayers = len(d_center)
        costh_fz = [[] for i in range(len(dlayers))]
        costh = np.empty(nvectors,dtype=float)
        
        args = (ids1,ids2,dlayers,costh,costh_fz)
        nframes = self.loop_trajectory('P2', args)

       
        costh2_mean = np.array([ np.array(c).mean() for c in costh_fz ])
        costh2_std  = np.array([ np.array(c).std()  for c in costh_fz ])
        s='P2_' +'-'.join(topol_vector)
        orientation = {'d':  d_center} 
        orientation.update({s: 1.5*costh2_mean-0.5, s+'(std)' : 1.5*costh2_std-0.5 })
        
        tf = perf_counter() - t0
        
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return orientation   
        

    def calc_conformation_density(self,dads,dmax,binl,option='__densityAndstats'):
        
        t0 = perf_counter()
        
        #initialize
        dlayers = self.get_layers(dads,dmax,binl)[1:]
        d_center = np.array([0.5*(b[0]+b[1]) for b in dlayers])
        
        stats = { k : 0 for k in ['adschains','train','looptailbridge',
                                  'tail','loop','bridge']}
        
        nlay = len(dlayers)
        dens = {k:np.zeros(nlay,dtype=float) for k in ['mtail','mloop','mbridge','ntail','nloop','nbridge'] }                
        dens.update({'d':d_center})
        
        #calculate
        args = (dads,dlayers, dens, stats)
        nframes = self.loop_trajectory('conformation'+option, args)
        
        #post_proc
        for k in ['ntail','nloop','nbridge']: dens[k] /= nframes
        for k in ['mtail','mloop','mbridge']: dens[k] *= 1.660539e-3/nframes
        dens.update({'d':d_center})
        
        for k in stats: stats[k] /= nframes
        for k in ['train','looptailbridge','tail','loop','bridge']: stats[k+'_perc'] = stats[k]/self.npol
        stats['adschains_perc'] = stats['adschains'] / len(self.chain_args)
            
        for k,v in stats.items():
            logger.info('{:s} = {:4.3f}'.format(k,v))
        
        tf = perf_counter() -t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return dens, stats
    
    def calc_particle_size(self):
        t0 = perf_counter()
        part_size = np.zeros(3)
        args = (part_size,)
        nframes = self.loop_trajectory('particle_size',args)
        part_size /= nframes
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return part_size
        
    def calc_dihedral_distribution(self,dads,dmax,binl):
        t0 = perf_counter()
        
        dlayers = self.get_layers(dads,dmax,binl)
        d_center = [0.5*(b[0]+b[1]) for b in dlayers]
        
        dih_types = self.dihedral_types
        dih_ids = self.dihedrals_ids_per_type # dictionary
        
        dih_distr = { d: {lay: [] for lay in dlayers} for d in dih_types }
        
        args = (dih_ids,dlayers,dih_distr)
        nframes = self.loop_trajectory('dihedral_distribution', args)
        
        dihedral_distr = {k:{lay:np.array(val) for lay,val in dih_distr[k].items()} for k in dih_distr }
        tf = perf_counter() -t0
       
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dihedral_distr
    
    def calc_chain_characteristics(self,dmin,dmax,binl):
        
        t0 = perf_counter()
        
        bins = np.arange(dmin,dmax,binl)
        dlayers = [(bins[i],bins[i+1]) for i in range(len(bins)-1)]
        d_center = [0.5*(b[0]+b[1]) for b in dlayers]
        nl = len(dlayers)
        
        chars_strlist = ['k2','Rg2','Rg','Ree2','asph','acyl', 'Rgxx_plus_yy', 'Rgxx_plus_yy', 'Rgyy_plus_zz']
        chars = {k:[[] for d in dlayers] for k in chars_strlist }

        chain_args = self.chain_args
        
        # calculate
        args  = chain_args,dlayers,chars
        nframes = self.loop_trajectory('chain_characteristics', args)
                
        #post_process
        chain_chars = {'d':np.array(d_center)}
        
        for k,v in chars.items():
            chain_chars[k] = np.array([ np.mean(chars[k][i]) for i in range(nl) ])
            chain_chars[k+'(std)'] = np.array([ np.std(chars[k][i]) for i in range(nl) ])
        
        tf= perf_counter() -t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return chain_chars
    
    ###### End of Main calculation Functions for structural properties ########
    
    def vects_per_t(self,ids1,ids2,
                         filters={},
                         dads=0):
        t0 = perf_counter()
        vec_t = dict()
        filt_per_t = dict()
        
        args = (ids1,ids2,filters,dads,vec_t,filt_per_t)
        
        nframes = self.loop_trajectory('vects_t', args)
                
        filt_per_t = rearrange_dict_keys(filt_per_t)
        tf = perf_counter()-t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return vec_t,filt_per_t

    def chains_CM(self,coords):
        chain_arg_keys = self.chain_args.keys()
        
        chain_cm = np.empty((len(chain_arg_keys),3),dtype=float)
        
        for i,args in enumerate(self.chain_args.values()):
            chain_cm[i] = CM(coords[args],self.atom_mass[args])
        
        return chain_cm

    def segs_CM(self,coords,segids):
        n = segids.shape[0]
        segcm = np.empty((n,3),dtype=float)
        numba_CM(coords,segids,self.atom_mass,segcm)
        return segcm
            

    def calc_dihedrals_t(self,dih_type,
                             dads=0,filters={'all':None}):
        t0 = perf_counter()
        
        dihedrals_t = dict()
        filt_per_t = dict()
       
        dih_ids = self.dihedrals_ids_per_type[dih_type] #array (ndihs,4)
        ids1 = dih_ids[:,0] ; ids2 = dih_ids[:,3] 

        args = (dih_ids,ids1,ids2,filters,dads,dihedrals_t,filt_per_t)
        
        nframes = self.loop_trajectory('dihedrals_t', args)
        
        filt_per_t = rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter()-t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return dihedrals_t,filt_per_t
    


    def calc_Ree_t(self,filters,
                       dads=0):
        t0 = perf_counter()
        #filters = {'chain_'+k: v for k,v in filters.items()}
        chain_is = []
        chain_ie = []
        for j,ch_args in self.chain_args.items():
            chain_is.append(ch_args[0])
            chain_ie.append(ch_args[-1])
        ids1 = np.array(chain_is)
        ids2 = np.array(chain_ie)
      
        Ree_t,filt_per_t = self.vects_per_t(ids1,ids2,
                                                     filters,
                                                     dads)
        
        tf = perf_counter()-t0
        
        print_time(tf,inspect.currentframe().f_code.co_name)
        #return {t:v[tot_filt] for t,v in dihedrals_t.items()}
        return Ree_t,filt_per_t
    
    def set_partial_charge(self):
        if not hasattr(self,'partial_charge'):
            self.charge_from_maps()
            charge = np.empty(self.at_types.shape[0],dtype=float)
            for i,ty in enumerate(self.at_types):
                charge[i] = self.charge_map[ty]
            self.partial_charge = charge
        return
    
    def calc_chain_dipole_moment_t(self,filters={'all':None},dads=1):
        t0 = perf_counter()
        
        filters = {'chain_'+k : v for k,v in filters.items()}
        
        self.set_partial_charge()

        
        dipoles_t = dict()
        filters_t = dict()
        args = (filters,dads,dipoles_t,filters_t)
        
        nframes = self.loop_trajectory('chain_dipole_moment',args)
        
        filters_t = rearrange_dict_keys(filters_t)
        
        tf = perf_counter() - t0
        
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dipoles_t,filters_t
    
    def find_segmental_ids(self,ids1,ids2,segbond):
        t0 = perf_counter()
        conn_cleaned = {k:v for k,v in self.connectivity.items()
                        if v != ('C','C')
                        }
        
        bond_ids = np.array(list(conn_cleaned.keys()))
        
        b0 = bond_ids[:,0]
        b1 = bond_ids[:,1]
        
        seg_ids = {i : list(np.arange(i1,i2+1,dtype='i')) 
                   for i,(i1,i2) in enumerate(zip (ids1,ids2))
                  }
        
        for i,ids in seg_ids.copy().items():
            f0 = np.isin(b0,ids)
            f1 = np.isin(b1,ids)
            seg_ids[i].extend(b1[f0])
            seg_ids[i].extend(b0[f1])
         
        seg_args = {k:np.unique(v) for k,v in seg_ids.items()}
        
        
        seg_ids_numpy = np.array(list(seg_args.values()))
        self.segmental_args = seg_ids_numpy
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return  seg_ids_numpy
    
    def calc_segmental_dipole_moment_t(self,topol_vector,
                                       segbond,filters={'all':None},
                                       dads=1):
        t0 = perf_counter()
        
        if not hasattr(self,'partial_charge'):
            charge = np.empty(self.at_types.shape[0],dtype=float)
            for i,ty in enumerate(self.at_types):
                charge[i] = self.charge_map[ty]
            self.partial_charge = charge
            
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        segmental_ids = self.find_segmental_ids(ids1, ids2, segbond)
        
        dipoles_t = dict()
        filters_t = dict()
        
        args = (filters,dads,ids1,ids2,segmental_ids,dipoles_t,filters_t)
        
        nframes = self.loop_trajectory('segmental_dipole_moment',args)
        
        filters_t = rearrange_dict_keys(filters_t)
        
        tf = perf_counter() - t0
        
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dipoles_t,filters_t
    
    def calc_segmental_dipole_moment_correlation(self,topol_vector,
                                       segbond,filters={'all':None},
                                       dads=1.025):
        t0 = perf_counter()
        dipoles_t,filters_t = self.calc_segmental_dipole_moment_t(topol_vector,
                                       segbond,filters,dads=1)
        
        ids1,ids2 = self.find_vector_ids(topol_vector)
        bond_distmatrix = self.bond_distance_matrix(ids1)
        
        unb = np.unique(bond_distmatrix)
        b0 = dict()
        b1 = dict()
        for k in unb:
            if k>0:
                b =  np.nonzero(bond_distmatrix ==k)
                b0[k] = b[0]
                b1[k] = b[1]
        
        correlations ={filt:{k:[] for k in unb if k>0} for filt in filters_t}
        
        args = (dipoles_t,filters_t,b0,b1,correlations)
        
        nframes = self.loop_trajectory('vector_correlations', args)
        
        for kf in filters_t:
            for k in correlations[kf]:
                c = correlations[kf][k]
                correlations[kf][k] = {'mean':np.mean(c),'std':np.std(c)}
                
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name)
    
        return correlations
    
    def calc_chainCM_t(self,filters={'all':None}, dads=1,option=''):
        t0 = perf_counter()
        filters = {'chain_'+k: v for k,v in filters.items()} #Need to modify when considering chains
        
        vec_t = dict()
        filt_per_t = dict()
        
        args = (filters,dads,vec_t,filt_per_t)
        
        nframes = self.loop_trajectory('chainCM_t'+option, args)
      
        filt_per_t = rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return vec_t,filt_per_t
    
    def calc_segCM_t(self,topol_vector,segbond,
                     filters={'all':None}, dads=1,option=''):
        t0 = perf_counter()
     
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        segmental_ids = self.find_segmental_ids(ids1, ids2, segbond)
        
        vec_t = dict()
        filt_per_t = dict()
        
        args = (filters,dads,ids1,ids2,segmental_ids,vec_t,filt_per_t)
        
        nframes = self.loop_trajectory('segCM_t'+option, args)
      
        filt_per_t = rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return vec_t,filt_per_t
    
    def calc_conformations_t(self,dads,option=''):
        t0 = perf_counter()
        
        confs_t = dict()
        args = (dads,confs_t)
        nframes = self.loop_trajectory('confs_t'+option, args)
        confs_t = rearrange_dict_keys(confs_t)
        
        tf = perf_counter() - t0
        print_time(tf,inspect.currentframe().f_code.co_name,nframes)
    
        return confs_t
    
    def calc_segmental_vectors_t(self,topol_vector,filters=None,
                                     dads=0):
       
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        t0 = perf_counter()
        
        segvec_t, filt_per_t = self.vects_per_t(ids1, ids2, 
                                                         filters,
                                                         dads)
        
        tf = perf_counter()-t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return segvec_t,filt_per_t
    
    def calc_adsorbed_segments_t(self,topol_vector,dads):
        
        t0 = perf_counter()
        
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        segvec_t, adsorbed_segments_t = self.vects_per_t(ids1, ids2,
                                                         filters={'space':(0,dads)})
        tf = perf_counter()-t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return adsorbed_segments_t[(0,dads)]
    
    def calc_adsorbed_chains_t(self,dads,filters={'adsorption':None}):
        t0 = perf_counter()
        
        chvec_t, adsorbed_chains_t = self.calc_chainCM_t(filters=filters,dads=dads)
        tf = perf_counter()-t0
        print_time(tf,inspect.currentframe().f_code.co_name)
        return adsorbed_chains_t['ads']
    

    
    def init_xt(self,xt,dtype=float):
        x0 =xt[0]
        nfr = len(xt)
        shape = (nfr,*x0.shape)
        x_nump = np.empty(shape,dtype=dtype)
        
        for i,t in enumerate(xt.keys()):
           x_nump[i] = xt[t]
        
        return  x_nump
    
    def init_prop(self,xt):
        nfr = len(xt)
        Prop_nump = np.zeros(nfr,dtype=float)
        nv = np.zeros(nfr,dtype=int)
        return Prop_nump,nv
  
    def get_inner_kernel_function(self,prop,filt_option,weights_t):
        mapper = {'P1':'costh'}
        name = mapper[prop]
        af_name = 'get'
        if filt_option is not None:
            ps = '_{:s}'.format(filt_option)
            name += ps 
            af_name+= ps
        if weights_t is not None:
            ps = '_weighted'
            name += ps
            af_name += ps
        func_name = '{:s}__kernel'.format(name)
        args_func_name = '{:s}__args'.format(af_name)
        logger.info('func name : {:s} , argsFunc name : {:s}'.format(func_name,args_func_name))
        funcs = (globals()[func_name],globals()[args_func_name])
        return funcs
  
    def Dynamics(self,prop,xt,filt_t=None,weights_t=None,
                 filt_option='simple', block_average=True):
        
        tinit = perf_counter()
        
        Prop_nump,nv = self.init_prop(xt)
        x_nump = self.init_xt(xt)
        
        if filt_t is not None:
            f_nump = self.init_xt(filt_t,dtype=bool)
            if filt_option is None:
                filt_option= 'simple'
        else:
            f_nump = None
            filt_option = None
        
        if weights_t is not None:
            w_nump = self.init_xt(weights_t)
        else:
            w_nump = None
        
        
        func,func_args = self.get_inner_kernel_function(prop,filt_option,weights_t)
        args = (func,func_args,
                Prop_nump,nv,
                x_nump,f_nump,w_nump,
                block_average)
        
        prop_kernel = globals()[prop+'_kernel']
        overheads = perf_counter() - tinit
        
        try:
            #prop_kernel(prop_nump, nv, x_nump, f_nump, nfr)
            prop_kernel(*args)
        except ZeroDivisionError as err:
            logger.error('Dynamics Run {:s} --> There is a {} --> Check your filters or weights'.format(prop,err))
            return None
        
       
        tf2 = perf_counter()
        if prop !='P2':
            dynamical_property = {t:p for t,p in zip(xt,args[2])}
        else:
            dynamical_property = {t:0.5*(3*p-1) for t,p in zip(xt,args[2])}
        tf3 = perf_counter() - tf2
        
        tf = perf_counter()-tinit
        #logger.info('Overhead: {:s} dynamics computing time --> {:.3e} sec'.format(prop,overheads+tf3))
        print_time(tf,inspect.currentframe().f_code.co_name +'" ---> Property: "{}'.format(prop))
        return dynamical_property

class Filters():
    def __init__(self):
        pass
    
    @staticmethod
    def calc_filters(self,filters,*args):
        bool_data = dict()
        #if filters is None: return bool_data
        for k,filt in filters.items():
            bool_data.update(  getattr(Filters,k)(self, filt, *args)  )
        return bool_data
    
    @staticmethod
    def all(self,filt,ids1,*args):
        return {'all':np.ones(ids1.shape[0],dtype=bool)}
    @staticmethod
    def chain_all(self,filt,*args):
        return {'all':np.ones(len(self.chain_args),dtype=bool)}
    @staticmethod
    def space(self,layers,ids1,ids2,coords,cm,*args):
        rm = 0.5*(coords[ids1] + coords[ids2])
        d = self.dfun(rm,cm)
        return Filters.filtLayers(layers,d)
    
    @staticmethod
    def filtLayers(layers,d):
        if iterable(layers):
            if iterable(layers[0]):
                return {dl : filt_uplow(d , dl[0], dl[1]) for dl in layers}
            else:
                return {layers: filt_uplow(d , layers[0], layers[1])}
        return dict()
    @staticmethod
    def filtLayers_inclucive(layers,d):
        if iterable(layers):
            if iterable(layers[0]):
                return {dl : filt_uplow_inclucive(d , dl[0], dl[1]) for dl in layers}
            else:
                return {layers: filt_uplow_inclucive(d , layers[0], layers[1])}
        return dict()

    @staticmethod
    def BondsTrainFrom(self,bondlayers,ids1,ids2,coords,cm,dads,box,*args):
        #t0 = perf_counter()
        #d = self.dfun(coords,cm)
        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations(dads,coords,box)
        
        args_rest_train = np.concatenate( (args_tail,args_loop,args_bridge ) )
        nbonds1 = self.ids_nbondsFrom_args(ids1,args_rest_train)
        nbonds2 = self.ids_nbondsFrom_args(ids2,args_rest_train)
        nbonds = np.minimum(nbonds1,nbonds2)
        
        return Filters.filtLayers_inclucive(bondlayers,nbonds)
    
    def BondsFromEndGroups(self,bondlayers,ids1,ids2,coords,cm,dads,box,*args):
        #t0 = perf_counter()
        args_endGroups = self.get_EndGroup_args()
        
        nbonds1 = self.ids_nbondsFrom_args(ids1,args_endGroups)
        nbonds2 = self.ids_nbondsFrom_args(ids2,args_endGroups)
        nbonds = np.minimum(nbonds1,nbonds2)
        
        return Filters.filtLayers_inclucive(bondlayers,nbonds)
    
    @staticmethod
    def BondsFromTrain(self,bondlayers,ids1,ids2,coords,cm,dads,box,*args):
        #t0 = perf_counter()
        #d = self.dfun(coords,cm)
        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations(dads,coords,box)

        nbonds1 = self.ids_nbondsFrom_args(ids1,args_train)
        nbonds2 = self.ids_nbondsFrom_args(ids2,args_train)
        nbonds = np.minimum(nbonds1,nbonds2)
        
        return Filters.filtLayers_inclucive(bondlayers,nbonds)

    @staticmethod
    def conformationDistribution(self,fconfs,ids1,ids2,coords,cm,dads,box,*args):
        
        #d = self.dfun(coords,cm)
        ads_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations(dads,coords,box)
       
        filt = dict()
        for conf,intervals in fconfs.items():
            
            args = locals()['args_'+conf]
            
            conf_chunks = self.conf_chunks(args)
            
            sizes = np.array([len(chunk) for chunk in conf_chunks])
            
            
            filt['{}:distr'.format(conf)] = sizes
            
            for inter in intervals:
                
                chunk_int =set()
                for chunk, size in zip(conf_chunks,sizes):
                    if inter[0]<=size<inter[1]:
                        chunk_int = chunk_int | chunk
                        
                args_chunk = np.array(list(chunk_int),dtype=int)
                f = Filters.filt_bothEndsIn(ids1, ids2, args_chunk)
                
                filt['{}:{}'.format(conf,inter)] = f
                
        return filt
    
    @staticmethod
    def conformations(self,fconfs,ids1,ids2,coords,cm,dads,box,*args):
        #rm = 0.5*(coords[ids1] + coords[ids2])
        # = perf_counter()
        #d = self.dfun(coords,cm)
        
        ads_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations(dads,coords,box)
        
        all_not_free = np.concatenate((args_train,args_tail,args_loop,args_bridge))
        
        all_args = self.all_args
        args_free = all_args [ np.logical_not( np.isin(all_args, all_not_free) ) ]

        filt = dict()
        if iterable(fconfs):
            for conf in fconfs:
                conf_args = locals()['args_'+conf]
                filt[conf] = Filters.filt_bothEndsIn(ids1, ids2, conf_args)
        elif type(fconfs) is str:
            conf_args = locals()['args_'+fconfs]
            filt[fconfs] = Filters.filt_bothEndsIn(ids1, ids2, conf_args)
        #print_time(perf_counter()-t0,inspect.currentframe().f_code.co_name)
        return filt
    
    @staticmethod
    def filt_bothEndsIn(ids1,ids2,args):
        f1 = np.isin(ids1, args)
        f2 = np.isin(ids2, args)
        return np.logical_and(f1,f2)
    

    @staticmethod
    def combine_filts(filts,filtc):
        filt_sc = dict()
        for s,fs in filts.items():
            for c,fc in filtc.items():
                filt_sc[(s,c)] = {k:np.logical_and(fs[k],fc[k]) for k in fs.keys()}
        return filt_sc
    
    @staticmethod
    def chain_space(self,layers,chain_cm,part_cm,*args):
        d = self.dfun(chain_cm,part_cm)
        return Filters.filtLayers(layers, d)
    
    @staticmethod
    def chain_characteristics(self,layers,chain_cm,part_cm,coords,*args):
        raise NotImplementedError('Filter on chain characteristics is not yet implemented')
    
    @staticmethod
    def get_chain_characteristic(self,char,chain_cm,coords):
        characteristic = np.empty(len(self.chain_args),dtype=float)
        
        for i,(j,ch_cm) in enumerate(zip(self.chain_args,chain_cm)):
            #find chain  center of mass
            c_ch = coords[self.chain_args[j]]
            at_mass_ch = self.atom_mass[self.chain_args[j]]
            #ch_cm = CM(c_ch,at_mass_ch)
            
            #compute gyration tensor,rg,asphericity,acylindricity,end-to-end-distance
            Sm = np.zeros((3,3),dtype=float)
            Ree2, Rg2, k2, asph, acyl, Rgxx_plus_yy, Rgxx_plus_zz, Rgyy_plus_zz \
            = chain_characteristics_kernel(c_ch, at_mass_ch,ch_cm,Sm)
            characteristic[i] = locals()[char]
            
        return characteristic
            
    @staticmethod
    def get_ads_degree(self,part_cm,coords,dads,*args):
        chain_arg_keys = self.chain_args.keys()
        cads = np.empty(len(chain_arg_keys),dtype=bool)
        degree = np.empty(cads.shape[0],dtype=float)
        
        for i,args in enumerate(self.chain_args.values()):
            d = self.dfun(coords[args], part_cm)
            f = filt_uplow(d, 0, dads)
            cads[i] = f.any()
            degree[i] = np.count_nonzero(f)/f.shape[0]
        
        return degree,cads
    @staticmethod
    def chain_adsorption(self,ads_degree, chain_cm, part_cm, coords, dads,*args):
        
        degree,cads = Filters.get_ads_degree(self,part_cm,coords,dads)
            
        filt_ads = dict()
        filt_ads.update( Filters.filtLayers(ads_degree,degree) )
        
        filt_ads.update({'ads':cads,'free':np.logical_not(cads)})
        
        return filt_ads
    def chain_weight_ads(self,dumb_value, chain_cm, part_cm, coords, dads,*args):
        
        degree,cads = Filters.get_ads_degree(self,part_cm,coords,dads)
        filt_ads = {'ads':cads,'free':np.logical_not(cads),'weights':degree} 
        
        return filt_ads
    
class coreFunctions():
    def __init__():
        pass
    @staticmethod
    def particle_size(self,frame,part_size):
        part_coords = self.get_coords(frame)[self.particle_filt]
        part_size += part_coords.max(axis = 0 ) - part_coords.min(axis = 0 )
        return 
    @staticmethod
    def theFilt(self,frame,filters,dads,ids1,ids2,filt_per_t):
        
        coords = self.get_whole_coords(frame)
        
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        cm = self.centerfun( CM( coords[self.particle_filt], 
                self.atom_mass[self.particle_filt]) )
        
        if frame == self.first_frame:
            self.time_zero=time
        
        key = self.get_timekey(time,self.time_zero)
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords,cm,dads,box)
    @staticmethod
    def theChainFilt(self,frame,filters,dads,filt_per_t):
        
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        part_cm = self.centerfun( CM( coords[self.particle_filt], 
                self.atom_mass[self.particle_filt]) )
        
        chain_cm = self.chains_CM(coords)
        
        if frame == self.first_frame:
            self.time_zero=time
        key = self.get_timekey(time,self.time_zero)
        
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, part_cm, coords, dads)
        
        return 
    @staticmethod
    def box_mean(self,frame,box):
        box+=self.get_box(frame)
        return
    @staticmethod
    def box_var(self,frame,box_var,box_mean_squared):
        box_var += self.get_box(frame)**2 - box_mean_squared
        return
    @staticmethod
    def vector_correlations(self,frame,vec_t,filt_t,bk0,bk1,correlation):
            
            
        timekey = self.get_time(frame) - self.get_time(self.first_frame)
        vec = vec_t[timekey]
        for kf in correlation:
            f = filt_t[kf][timekey]
            for k in correlation[kf]:
              #  t0 = perf_counter()
                b0 = bk0[k]
                b1 = bk1[k]
                f01 = np.logical_and(f[b0],f[b1])
                v0 = vec[b0][f01]
                v1 = vec[b1][f01]
                
               # tf = perf_counter()
              #  print_time(tf-t0,'{}:{} :manipulating data'.format(kf,k))
                try:
                    costh = costh__parallelkernel(v0,v1)
                    correlation[kf][k].append( costh )
                except ZeroDivisionError:
              #      logger.warning('In frame {:d} --> For {} and bond distance {:d} there are no statistics'.format(frame,kf,k))
                    pass
            #    tf2 = perf_counter()
             #   print_time(tf2-tf,'{}:{} :computationsa'.format(kf,k))
        return
                
    @staticmethod
    def segmental_dipole_moment(self,frame,filters,dads,ids1,ids2,
                                segment_args,dipoles_t,filt_per_t):
        
            
        coords = self.get_whole_coords(frame)
        #coords = self.get_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        cm = self.centerfun( CM( coords[self.particle_filt], 
                self.atom_mass[self.particle_filt]) )

        if frame == self.first_frame:
            self.time_zero=time
        
        key = self.get_timekey(time,self.time_zero)
        
        n = segment_args.shape[0]
        
        dipoles = np.empty((n,3),dtype=float)
        pc = self.partial_charge.reshape((self.partial_charge.shape[0],1))
        
        numba_dipoles(pc,coords,segment_args,dipoles)
        
        dipoles_t[key] = dipoles  
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords,cm,dads,box)
        
        return
    @staticmethod
    def chain_dipole_moment(self,frame,filters,dads,dipoles_t,filt_per_t):
        
            
        coords = self.get_whole_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        part_cm = self.centerfun( CM( coords[self.particle_filt], 
                self.atom_mass[self.particle_filt]) )
        
        chain_cm = self.chains_CM(coords)
        
        
        if frame == self.first_frame:
            self.time_zero=time
        
        key = self.get_timekey(time,self.time_zero)
        
        n = chain_cm.shape[0]
        
        dipoles = np.empty((n,3),dtype=float)
        pc = self.partial_charge.reshape((self.partial_charge.shape[0],1))
        for i,(j, cargs) in enumerate(self.chain_args.items()):
            dipoles[i] = np.sum(pc[cargs]*coords[cargs],axis=0)
        
        dipoles_t[key] = dipoles  
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, part_cm, coords, dads)
        
        return
    
    @staticmethod
    def mass_density_profile__pertype(self,frame,nbins,bins,
                                  rho,rho_per_atom_type,ftt,mass_pol):
        

        coords,box,d,cs = self.get_frame_essentials(frame)
        d = d[self.pol_filt]
        
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(box,bins[i],bins[i+1])
            fin_bin = filt_uplow(d, bins[i], bins[i+1])
            rho[i] += numba_sum(mass_pol[fin_bin])/vol_bin
  
            for t in rho_per_atom_type.keys():
                ft = np.logical_and( fin_bin,ftt[t])
                rho_per_atom_type[t][i] += numba_sum(mass_pol[ft])/vol_bin
        return
    
    @staticmethod
    def mass_density_profile(self,frame,nbins,bins,
                                  rho,mass_pol):
        
        coords,box,d,cs = self.get_frame_essentials(frame)
        d = d[self.pol_filt]
        #2) Caclulate profile
        vol_bin = self.volfun(box,bins[0],bins[1])
        
        for i in range(nbins):     
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += np.sum(mass_pol[fin_bin])/vol_bin
        return
    
    @staticmethod
    def mass_density_profile_flux(self,frame,nbins,bins,
                                  rho,mass_pol,rho_mean_sq):
        
        coords,box,d,cs = self.get_frame_essentials(frame)
        d = d[self.pol_filt]
        #2) Caclulate profile
        vol_bin = self.volfun(box,bins[0],bins[1])
        
        for i in range(nbins):     
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += (np.sum(mass_pol[fin_bin])/vol_bin)**2-rho_mean_sq[i]
        return
    
    @staticmethod
    def mass_density_profile__2side(self,frame,nbins,bins,
                                  rho_up,rho_down,mass_pol):
        
        coords = self.frame_coords(frame)
        box = self.get_box(frame)
        cs = self.centerfun(CM( coords[self.particle_filt], 
                               self.atom_mass[self.particle_filt]))
        
        dfun = getattr(Distance_Functions,self.conftype +'__2side')
        d = dfun(coords[self.pol_filt],cs) 
        vol_bin = self.volfun(box,bins[0],bins[1])*0.5 # needed because in volfun the volume of each bin is multiplied by 2
        #2) Caclulate profile
        for i in range(nbins):    
            fin_bin_up =   filt_uplow(d,bins[i],bins[i+1])
            fin_bin_down = filt_uplow(d,-bins[i+1],-bins[i])
            rho_up[i] += np.sum(mass_pol[fin_bin_up])/vol_bin
            rho_down[i] += np.sum(mass_pol[fin_bin_down])/vol_bin
            
        return
    
    @staticmethod
    def number_density_profile__pertype(self,frame,nbins,bins,
                                  rho,rho_per_atom_type,ftt):
        
        coords,box,d,cs = self.get_frame_essentials(frame)
        d = d[self.pol_filt]
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(box,bins[i],bins[i+1])
            fin_bin = filt_uplow(d, bins[i], bins[i+1])
            rho[i] += np.count_nonzero(fin_bin)/vol_bin
  
            for t in rho_per_atom_type.keys():
                ft = np.logical_and( fin_bin,ftt[t])
                rho_per_atom_type[t][i] += np.count_nonzero(ft)/vol_bin
    
    @staticmethod
    def number_density_profile(self,frame,nbins,bins,rho):
        
        coords,box,d,cs = self.get_frame_essentials(frame)
        d = d[self.pol_filt]
        
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(box,bins[i],bins[i+1])
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += np.count_nonzero(fin_bin)/vol_bin
        return
  
    @staticmethod
    def conformation__densityAndstats(self,frame,dads,dlayers,dens,stats):                
        
        coords = self.get_whole_coords(frame)
        box = self.get_box(frame)
        #1) ads_chains, trains,tails,loops,bridges
        ads_chains, args_train, args_tail, args_loop, args_bridge = self.conformations( dads,coords,box)
        
        #check_occurances(np.concatenate((args_train,args_tail,args_bridge,args_loop)))
        
        coreFunctions.conformation_dens(self,frame, dlayers, dens,
                                           ads_chains, args_train, args_tail,
                                           args_loop, args_bridge)
        
        coreFunctions.conformation_stats(stats,ads_chains, args_train, args_tail, 
                             args_loop, args_bridge)
        return
    
    @staticmethod
    def conformation__density(self,frame,dads,dlayers,dens,stats):                
        
        coords = self.get_whole_coords(frame)
        box = self.get_box(frame)
        #1) ads_chains, trains,tails,loops,bridges
        ads_chains, args_train, args_tail, args_loop, args_bridge = self.conformations( dads,coords,box)
        
        #check_occurances(np.concatenate((args_train,args_tail,args_bridge,args_loop)))
        
        coreFunctions.conformation_dens(self,frame, dlayers, dens,
                                           ads_chains, args_train, args_tail,
                                           args_loop, args_bridge)
        
        return

    @staticmethod
    def conformation__stats(self,frame,dads,dlayers,dens,stats):                
        
        coords = self.get_whole_coords(frame)
        box = self.get_box(frame)
        #1) ads_chains, trains,tails,loops,bridges
        ads_chains, args_train, args_tail, args_loop, args_bridge = self.conformations( dads,coords,box)
        
        #check_occurances(np.concatenate((args_train,args_tail,args_bridge,args_loop)))
        
        
        coreFunctions.conformation_stats(stats,ads_chains, args_train, args_tail, 
                             args_loop, args_bridge)
        return
    
    @staticmethod
    def conformation_dens(self,frame, dlayers,dens,
                             ads_chains, args_train, args_tail, 
                             args_loop, args_bridge):
        
        coords = self.get_whole_coords(frame)
        box = self.get_box(frame)
        d = self.get_minimum_distance_from_particle(coords,box)
        
        
        d_tail = d[args_tail]
        d_loop = d[args_loop]          
        d_bridge = d[args_bridge]
        
        for l,dl in enumerate(dlayers):
            args_tl = args_tail[filt_uplow(d_tail, dl[0], dl[1])]
            args_lp = args_loop[filt_uplow(d_loop, dl[0], dl[1])]
            args_br =  args_bridge[filt_uplow(d_bridge, dl[0], dl[1])]
            
            vol_bin=self.volfun(box,dl[0],dl[1])
            
            dens['ntail'][l] += args_tl.shape[0]/vol_bin
            dens['nloop'][l] += args_lp.shape[0]/vol_bin
            dens['nbridge'][l] +=args_br.shape[0]/vol_bin
            
            
            dens['mtail'][l] += numba_sum(self.atom_mass[args_tl])/vol_bin
            dens['mloop'][l] += numba_sum(self.atom_mass[args_lp])/vol_bin
            dens['mbridge'][l] += numba_sum(self.atom_mass[args_br])/vol_bin
        return    

    @staticmethod
    def conformation_stats(stats,ads_chains, args_train, args_tail, 
                             args_loop, args_bridge):
        stats['train'] += args_train.shape[0]
        stats['adschains']+= ads_chains.shape[0] 
        stats['looptailbridge']+= (args_loop.shape[0]+args_tail.shape[0]+args_bridge.shape[0])
        stats['tail']+=args_tail.shape[0]
        stats['loop']+=args_loop.shape[0]
        stats['bridge']+=args_bridge.shape[0]
        return 
    
    @staticmethod
    def dihedral_distribution(self,frame,dih_ids,dlayers,dih_distr):
        
        box = self.get_box(frame)
        coords = self.get_whole_coords(frame)
        cs = self.centerfun(CM( coords[self.particle_filt], self.atom_mass[self.particle_filt]))

        for k,d_ids in dih_ids.items():
            rm = 0.5*( coords[d_ids[:,1]] + coords[d_ids[:,2]] )
            d = self.dfun(rm,cs)
            dih_val = np.empty(d_ids.shape[0],dtype=float)
            dihedral_distribution_kernel(d_ids,coords,dih_val)
            for lay in dlayers:
                fin_bin = filt_uplow(d, lay[0], lay[1])
                dih_distr[k][lay].extend(dih_val[fin_bin])
        return
    
    @staticmethod
    def P2(self,frame,ids1,ids2,dlayers,costh,costh_fz):
        #1) coords
        coords = self.get_whole_coords(frame)
        box =self.get_box(frame)

        #2) calc_particle_cm
        cs = self.centerfun(CM( coords[self.particle_filt], self.atom_mass[self.particle_filt]))
        
        r1 = coords[ids1]; r2 = coords[ids2]
        
        rm = 0.5*(r1+r2)
        d = self.dfun(rm, cs)
        uv = self.unit_vectorFun(rm,cs)
        
        costhsquare__kernel(costh,r2-r1,uv)
        
        for i,dl in enumerate(dlayers):
            filt = filt_uplow(d, dl[0], dl[1])
            costh_fz[i].extend(costh[filt])
        
        return   
    
    @staticmethod
    def chain_characteristics(self,frame,chain_args,dlayers,chars):
        #1) translate the system
        
        coords = self.get_whole_coords(frame)
        box = self.get_box(frame)
        cs = self.centerfun(CM( coords[self.particle_filt], self.atom_mass[self.particle_filt]))
                   
        for j in chain_args:
            #find chain  center of mass
            c_ch = coords[chain_args[j]]
            at_mass_ch = self.atom_mass[chain_args[j]]
            ch_cm = CM(c_ch,at_mass_ch)
            
            #compute gyration tensor,rg,asphericity,acylindricity,end-to-end-distance
            Sm = np.zeros((3,3),dtype=float)
            Ree2, Rg2, k2, asph, acyl, Rgxx_plus_yy, Rgxx_plus_zz, Rgyy_plus_zz \
            = chain_characteristics_kernel(c_ch, at_mass_ch,ch_cm,Sm)
            
            Rg = Rg2**0.5
            local_dict = locals()
            #Assign values
            d = self.dfun(ch_cm.reshape((1,3)),cs) #scalar
            for i,dl in enumerate(dlayers):
                if dl[0]< d[0] <=dl[1]:
                    for char in chars:
                        chars[char][i].append(local_dict[char])
                    break
        return
    
    @staticmethod
    def dihedrals_t(self,frame,
                dih_ids, ids1, ids2, filters, dads, dihedrals_t, filt_per_t):
        
        t0 = perf_counter()
        
        coords = self.get_whole_coords(frame)
        cm = self.centerfun( CM( coords[self.particle_filt], 
                               self.atom_mass[self.particle_filt]) )
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        dih_val = np.empty(dih_ids.shape[0],dtype=float) # alloc 
        dihedral_distribution_kernel(dih_ids,coords,dih_val)

        if frame == self.first_frame :
            self.time_zero = time
        key = self.get_timekey(time, self.time_zero)
        dihedrals_t[key] = dih_val
        
        del dih_val #deallocating for safty
        tm = perf_counter()
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords,cm,dads,box)
        tf = perf_counter()
        if frame ==1:
            logger.info('Dihedrals_as_t: Estimate time consuption --> Main: {:2.1f} %, Filters: {:2.1f} %'.format((tm-t0)*100/(tf-t0),(tf-tm)*100/(tf-t0)))
        return

    @staticmethod
    def vects_t(self,frame,ids1,ids2,filters,dads,vec_t,filt_per_t):
       
        coords = self.get_whole_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        cm = self.centerfun( CM( coords[self.particle_filt], 
                self.atom_mass[self.particle_filt]) )
        
        vec = coords[ids2,:] - coords[ids1,:]
        
        if frame == self.first_frame:
            self.time_zero = time
            
        key = self.get_timekey(time,self.time_zero)
        
        vec_t[key] = vec
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            ids1,ids2,coords,cm,dads,box)
        return     
    
    @staticmethod
    def confs_t(self,frame,dads,confs_t):
        
        coords,box,d,cs = self.get_unwrappedframe_essentials(frame)
        time = self.get_time(frame)
        
        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations(dads,coords,box)
        x = dict()
        ntot = args_train.shape[0] + args_tail.shape[0] +\
               args_loop.shape[0] + args_bridge.shape[0]
        for k in ['train','tail','loop','bridge']:
            args = locals()['args_'+k]
            x[k] = args.shape[0]/ntot
        x['ads_chains'] = ads_chains.shape[0]/len(self.chain_args)
        
        if frame == self.first_frame:
            self.time_zero = time
            
        key = self.get_timekey(time,self.time_zero)
        
        confs_t[key] = x
        
        return
    @staticmethod
    def confs_t__perchain(self,frame,dads,confs_t):
        
        coords,box,d,cs = self.get_unwrappedframe_essentials(frame)
        time = self.get_time(frame)
        
        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations(dads,coords,box)
        x = dict()
        for k in ['train','tail','loop','bridge']:
            args = locals()['args_'+k]
            x[k] =  [ np.count_nonzero( np.isin(a, args ) )/a.shape[0] 
                                      for a in self.chain_args.values() ] 
                                    
        
        if frame == self.first_frame:
            self.time_zero = time
            
        key = self.get_timekey(time,self.time_zero)
        
        confs_t[key] = x
        
        return
    
    @staticmethod
    def chainCM_t(self,frame,filters,dads,vec_t,filt_per_t):
        
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        part_cm = self.centerfun( CM( coords[self.particle_filt], 
                self.atom_mass[self.particle_filt]) )
        
        chain_cm = self.chains_CM(coords)
        
        if frame == self.first_frame:
            self.time_zero=time
        key = self.get_timekey(time,self.time_zero)
        
        vec_t[key] =  chain_cm
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, part_cm, coords, dads)
        
        return 
    
    @staticmethod
    def segCM_t(self,frame,filters,dads,
                  ids1,ids2,segment_ids,
                  vec_t,filt_per_t):
        
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        coords_whole = self.get_whole_coords(frame)
        
        part_cm = self.centerfun( CM( coords_whole[self.particle_filt], 
                self.atom_mass[self.particle_filt]) )
        
        seg_cm = self.segs_CM(coords,segment_ids)
        
        if frame == self.first_frame:
            self.time_zero=time
        key = self.get_timekey(time,self.time_zero)
        
        vec_t[key] =  seg_cm
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            ids1,ids2,coords_whole,part_cm,dads,box)
        
        return 
    
    @staticmethod
    def gofr(self,frame,bins,gofr):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        
        n = coords.shape[0]
        pd = np.empty(int(n*(n-1)/2),dtype=float)

        pair_dists(coords,box,pd)
        numba_bin_count(pd,bins,gofr)
        

        return
    @staticmethod
    def gofr_type(self,frame,fty,bins,gofr):
        
        coords = self.get_coords(frame)[fty]
        box = self.get_box(frame)
        
        n = coords.shape[0]
        pd = np.empty(int(n*(n-1)/2),dtype=float)
 
        pair_dists(coords,box,pd)

        numba_bin_count(pd,bins,gofr)
        return
    @staticmethod
    def gofr_type_to_all(self,frame,fty,bins,gofr):
        
        coords = self.get_coords(frame)
        
        coords_ty = coords[fty]
        box = self.get_box(frame)
        
        n = coords.shape[0]
        nty = coords_ty.shape[0]
        pd = np.empty(int(n*nty),dtype=float)
        
        pair_dists_general(coords,coords_ty,box,pd)
        
        numba_bin_count(pd,bins,gofr)

        return
    
    @staticmethod
    def gofr_type_to_type(self,frame,fty1,fty2,bins,gofr):
        
        coords = self.get_coords(frame)
        
        coords_ty1 = coords[fty1]
        coords_ty2 = coords[fty2]
        box = self.get_box(frame)
        
        nty1 = coords_ty1.shape[0]
        nty2 = coords_ty2.shape[0]
        pd = np.empty(int(nty1*nty2),dtype=float)

        pair_dists_general(coords_ty1,coords_ty2,box,pd)
        numba_bin_count(pd,bins,gofr)

        return
    


@jit(nopython=True,fastmath=True)
def fill_property(prop,nv,i,j,value,mi,block_average):
    
    idx = j-i
    if block_average:
        try:
            prop[idx] +=  value/mi
            nv[idx] += 1
        except:
            pass
    else:
        prop[idx] +=  value
        nv[idx] += mi
    
    return



@jit(nopython=True,fastmath=True,parallel=True)
def P1_kernel(func,func_args,P1,nv,xt,ft=None,wt=None,
              block_average=False):
    
    n = xt.shape[0]

    #func = costh__kernel_simple
    
    for i in range(n):
        for j in prange(i,n):
            
            args = func_args(i,j,xt,ft,wt)
            
            value,mi = func(*args)
            
            fill_property(P1,nv,i,j,value,mi,block_average)
        
    for i in prange(n):    
        P1[i] /= nv[i]
    return 

@jit(nopython=True,fastmath=True)
def get__args(i,j,xt,ft,wt):
    return (xt[i],xt[j])

@jit(nopython=True,fastmath=True)
def get_weighted__args(i,j,xt,ft,wt):
    return (xt[i], xt[j],  wt[i])


@jit(nopython=True,fastmath=True)
def get_simple__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i])

@jit(nopython=True,fastmath=True)
def get_simple_weighted__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i],wt[i])


@jit(nopython=True,fastmath=True)
def get_strict__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i],ft[j])

@jit(nopython=True,fastmath=True)
def get_strict_weighted__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i],ft[j],wt[i])


@jit(nopython=True,fastmath=True)
def get_change__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i],ft[j])

@jit(nopython=True,fastmath=True)
def get_change_weighted__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i],ft[j],wt[i])

@jit(nopython=True,fastmath=True)
def costh__kernel(r1,r2):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        costh = costh_kernel(r1[i],r2[i])
        tot+=costh
        mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def costh_simple__kernel(r1,r2,ft0):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=costh
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def costh_strict__kernel(r1,r2,ft0,fte):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and fte[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=costh
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def costh_change__kernel(r1,r2,ft0,fte):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and not fte[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=costh
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def costh_weighted__kernel(r1,r2,w):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        costh = costh_kernel(r1[i],r2[i])
        tot+=w[i]*costh
        mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def costh_simple_weighted__kernel(r1,r2,ft0,w):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=w[i]*costh
            mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def costh_strict_weighted__kernel(r1,r2,ft0,fte,w):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and fte[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=w[i]*costh
            mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def costh_change_weighted__kernel(r1,r2,ft0,fte,w):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and not fte[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=w[i]*costh
            mi+=w[i]
    return tot,mi


@jit(nopython=True,fastmath=True,parallel=True)
def P2_kernel(P2,nv,x,ft,n):
    for i in range(n):
        ft0 = ft[i]   
        x0 = x[i]
        for j in prange(i,n):
            try:
                value = cos2th__kernel_with_filter(x0,x[j],ft0)
            except:
                pass
            else:
                idx = j-i
                P2[idx] +=  value
                nv[idx] += 1
        
    for i in prange(n):    
        P2[i] /= nv[i]
    return 

@jit(nopython=True,parallel=True)
def numba_bin_count(d,bins,counter):
    for j in prange(bins.shape[0]-1):
        for i in range(d.shape[0]):
            if bins[j]<d[i] and d[i] <=bins[j+1]:
                counter[j] +=1
    return
@jit(nopython=True,fastmath=True,parallel=True)
def MSD_kernel(msd,nv,x,ft,n):
    for i in range(n):
        ft0= ft[i]
        x0 = x[i]
        for j in prange(i,n):
            try:
                Rt = x0-x[j]
                value = mean_norm_square(Rt,ft0)
            except:
                pass
            else:
                idx = j-i
                msd[idx] +=  value
                nv[idx] += 1
        
    for i in range(n):    
        msd[i] /= nv[i]
    return 


@jit(nopython=True,fastmath=True)
def norm_square(x1,x2):
    nm = 0
    for i in range(x1.shape[0]):
        nm+= x1[i]*x2[i]
    return nm

@jit(nopython=True,fastmath=True)
def mean_norm_square(Rt,ft0):
    mu = 0
    ni = 0
    for i in range(Rt.shape[0]):
        if ft0[i]:
             mu += norm_square(Rt[i],Rt[i])
             ni+=1
    return mu/ni
@jit(nopython=True,fastmath=True)
def mean_norm_square__weighted(Rt,ft0,w):
    mu = 0
    ni = 0
    for i in range(Rt.shape[0]):
        if ft0[i]:
             mu += w[i]*norm_square(Rt[i],Rt[i])
             ni+=w[i]
    return mu/ni

@jit(nopython=True,fastmath=True,parallel=True)
def tacf_kernel(tacf,nv,x,ft,n):
    for i in range(n):
        ft0 = ft[i]
        x0 = x[i]
        mu = mean_wfilt_kernel(x0,ft0)
        
        mu_square = mu**2
        var = secmoment_wfilt_kernel(x0, ft0) - mu_square
        
        for j in prange(i,n):
            try:
                value = covariance_wfilter_kernel(x0,x[j],ft0)
            except:
                pass
            else:
                idx = j-i               
                tacf[idx] +=  (value - mu_square)/var
                nv[idx] += 1
        
    for i in prange(n):    
        tacf[i] /= nv[i]
    return

@jit(nopython=True,fastmath=True,parallel=True)
def DK_kernel(phi,nv,x,n):
    for i in range(n):
        x0 = x[i]
        for j in prange(i,n):
            try:
                xt = x[j]
                value = DesorptionCorrelation(x0,xt)
            except:
                pass
            else:
                idx = j-i
                phi[idx] +=  value
                nv[idx] += 1
        
    for i in prange(n):    
        phi[i] /= nv[i]
    return 


@jit(nopython=True,fastmath=True)
def DesorptionCorrelation(x0,xt):
    value = 0 ; m = 0
    for k1 in range(x0.shape[0]):
        if x0[k1]:
            if xt[k1]:
                value+=1
            else:
                value+=0
            m+=1
    value/=m
    return value

@jit(nopython=True,fastmath=True)
def DesorptionCorrelation_weighted(x0,xt,w):
    value = 0 ; m = 0
    for k1 in range(x0.shape[0]):
        if x0[k1]:
            if xt[k1]:
                value += w[k1]
            else:
                value+=0
            m+=w[k1]
    value/=m
    return value
@jit(nopython = True,fastmath=True)
def mean_wfilt_kernel(x,f):
    m =0 ; mi=0
    for i in range(x.shape[0]):
        if f[i]:
            m+=x[i]
            mi+=1
    m/=mi
    return m
@jit(nopython=True,fastmath=True)
def covariance_wfilter_kernel(x0,xt,ft0):
    m=0 ; mi =0
    for i in range(x0.shape[0]):
        if ft0[i]:
            m+=xt[i]*x0[i]
            mi+=1
    m/=mi
    return m 
@jit(nopython = True,fastmath=True)
def secmoment_wfilt_kernel(x,f):
    se=0 ; mi=0
    for i in range(x.shape[0]):
        if f[i]:
            se+=x[i]**2
            mi+=1
    se /=mi
    return se 
@jit(nopython = True,fastmaeth=True)
def var_wfilt_kernel(x,f):
    var = secmoment_wfilt_kernel(x,f) - mean_wfilt_kernel(x,f)**2
    return var


@jit(nopython=True,fastmath=True,parallel=True)
def costh__parallelkernel_with_filter_strict(r1,r2,ft0,ftt):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and ftt[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=costh
            mi+=1
    ave = tot/mi
    return ave


@jit(nopython=True,fastmath=True)#,parallel=True)
def costh__kernel_with_filter_strict(r1,r2,ft0,ftt):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and ftt[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=costh
            mi+=1
    ave = tot/mi
    return ave

@jit(nopython=True,fastmath=True)#,parallel=True)
def costh__kernel_with_filter_change(r1,r2,ft0,ftt):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and not ftt[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=costh
            mi+=1
    ave = tot/mi
    return ave

@jit(nopython=True,fastmath=True)#,parallel=True)
def costh__kernel_with_filter(r1,r2,ft0,ft2=None):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=costh
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)#,parallel=True)
def costh__kernel_with_filter_weighted(r1,r2,ft0,w):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=w[i]*costh
            mi+=w[i]
    ave = tot/mi
    return ave

@jit(nopython=True,fastmath=True)#,parallel=True)
def cos2th__kernel_with_filter(r1,r2,ft0):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot+=costh*costh
            mi+=1
    ave = tot/mi
    return ave
@jit(nopython=True,fastmath=True)
def cos2th__kernel_with_filter_weighted(r1,r2,ft0,w):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            costh = costh_kernel(r1[i],r2[i])
            tot += w[i]*costh*costh
            mi  += w[i]
    ave = tot/mi
    return ave
@jit(nopython=True,fastmath=True)
def costh_kernel(r1,r2):
    costh=0 ; rn1 =0 ;rn2=0
    n = r1.shape[0]
    for j in range(n):
        costh+=r1[j]*r2[j]
        rn1+=r1[j]*r1[j]
        rn2+=r2[j]*r2[j]
    rn1 = rn1**0.5
    rn2 = rn2**0.5
    costh/=rn1*rn2
    return costh

@jit(nopython=True,fastmath=True,parallel=True)
def costh__parallelkernel(r1,r2):
    tot = 0
    for i in prange(r1.shape[0]):
        tot += costh_kernel(r1[i],r2[i])
    ave = tot/float(r1.shape[0])
    return ave



@jit(nopython=True,fastmath=True)
def costhsquare__kernel(costh,r1,r2):
    for i in range(r1.shape[0]):
        costh[i] = costh_kernel(r1[i],r2[i])**2

@jit(nopython=True,fastmath=True)
def costhmean__kernel(r1,r2):
    mean =0
    for i in range(r1.shape[0]):
        costh = costh_kernel(r1[i],r2[i])
        mean+=costh    
    ave = mean/r1.shape[0]
    return ave



@jit(nopython=True,fastmath=True)
def unwrap_coords_kernel(unc,k0,k1,b2,n,dim,box):
    for j in dim:
        for i in n:
            if unc[k0[i],j] - unc[k1[i],j] > b2[j]:
                unc[k1[i],j] += box[j]
            elif unc[k1[i],j] - unc[k0[i],j] > b2[j]:
                unc[k1[i],j] -= box[j]
    return unc

@jit(nopython=True,fastmath=True)
def end_to_end_distance(coords):
    rel = coords[0]-coords[coords.shape[0]-1]
    Ree2 = np.dot(rel,rel)
    return Ree2
@jit(nopython=True,fastmath=True)
def chain_characteristics_kernel(coords,mass,ch_cm,Gyt):
    rel = coords[0]-coords[coords.shape[0]-1]
    Ree2 = np.dot(rel,rel)
    rccm = coords-ch_cm   
    for i in range(rccm.shape[0]):
        Gyt+=mass[i]*np.outer(rccm[i],rccm[i])
    Gyt/=np.sum(mass)
    S =  np.linalg.eigvals(Gyt)
    S =-np.sort(-S) # sorting in desceanding order
    Rg2 =np.sum(S)
    Shat = Gyt-Rg2*np.identity(3)/3
    asph = S[0] -0.5*(S[1]+S[2])
    acyl = S[1]-S[2]
    k2 = (asph**2 + 0.75*acyl**2)/Rg2**2
    
    Rgxx_plus_yy = Gyt[0][0] + Gyt[1][1]
    Rgxx_plus_zz = Gyt[0][0] + Gyt[2][2]
    Rgyy_plus_zz = Gyt[1][1] + Gyt[2][2]
    
    return Ree2, Rg2,k2,asph,acyl, Rgxx_plus_yy, Rgxx_plus_zz, Rgyy_plus_zz
   
@jit(nopython=True, fastmath=True,parallel=True)
def dihedral_distribution_kernel(dih_ids,coords,dih_val):
    for i in prange(dih_ids.shape[0]):
        r0 = coords[dih_ids[i,0]]
        r1 = coords[dih_ids[i,1]]  
        r2 = coords[dih_ids[i,2]]
        r3 = coords[dih_ids[i,3]]
        dih_val[i] = calc_dihedral(r0,r1,r2,r3)
    return 

@jit(nopython=True,fastmath=True)
def numba_sum_wfilt(x,filt):
    s = 0
    for i in range(x.shape[0]):
        if filt[i]: 
            s += x[i]
    return s

@jit(nopython=True,fastmath=True,parallel=True)
def numba_parallel_sum_wfilt(x,filt):
    s = 0
    for i in prange(x.shape[0]):
        if filt[i]: 
            s += x[i]
    return s

@jit(nopython=True,fastmath=True)
def numba_sum(x):
    s =0 
    for i in range(x.shape[0]):
        s+=x[i]
    return s

@jit(nopython=True,fastmath=True,parallel=True)
def numba_parallel_sum(x):
    s =0 
    for i in prange(x.shape[0]):
        s+=x[i]
    return s


class Analysis_Crystals(Analysis):
    def __init__(self,molecular_system,memory_demanding=True,topol=True):
        super().__init__(molecular_system,memory_demanding)
        self.topol =True
        return
    

def center_of_bins(bins):
    nbins = bins.shape[0] - 1
    r = [0.5*(bins[i]+bins[i+1]) for i in range(nbins)]
    return np.array(r)

def filt_uplow(x,yl,yup):
    return np.logical_and(np.greater(x,yl),np.less_equal(x,yup))

def filt_uplow_inclucive(x,yl,yup):
    return np.logical_and(np.greater_equal(x,yl),np.less_equal(x,yup))

def binning(x,bins):
    nbins = bins.shape[0]-1
    n_in_bins = np.empty(nbins,dtype=int) 
    for i in np.arange(0,nbins,1,dtype=int):
        filt = filt_uplow(x,bins[i],bins[i+1])
        n_in_bins[i] = np.count_nonzero(filt)
    return n_in_bins


@jit(nopython=True,fastmath=True)
def minimum_image(relative_coords,box):
    imaged_rel_coords = relative_coords.copy()
    for i in prange(relative_coords.shape[0]):
        for j in range(3):
            if relative_coords[i][j] > 0.5*box[j]:
                imaged_rel_coords[i][j] -= box[j]
            elif relative_coords[i][j] < -0.5*box[j]:
                imaged_rel_coords[i][j] += box[j]  
    return imaged_rel_coords
@jit(nopython=True,fastmath=True,parallel=True)
def pair_dists(coords,box,dists):
    n = coords.shape[0]

    for i in prange(n):
        rel_coords = coords[i] - coords[i+1:]
        rc = minimum_image(rel_coords,box)
        dist = np.sum(rc*rc,axis=1)**0.5
        idx_i = i*n
        for k in range(0,i+1):
            idx_i-=k
        for j in range(rc.shape[0]):
            dists[idx_i+j] = dist[j]
    
    return
@jit(nopython=True,fastmath=True,parallel=True)
def pair_dists_general(coords1,coords2,box,dists):
    n1 = coords1.shape[0]
    n2 = coords2.shape[0] 
    for i in prange(n1):
        rel_coords = coords1[i] - coords2
        rc = minimum_image(rel_coords,box)
        dist = np.sum(rc*rc,axis=1)**0.5
        for j in range(n2):
            dists[i*n2+j] = dist[j]
    return

@jit(nopython=True,fastmath=True)
def CM(coords,mass):
    cm = np.sum(mass*coords.T,axis=1)/mass.sum()
    return cm
#@jit(nopython=True,fastmath=True)
def implement_pbc(coords,boxsize,box='rectangular'):
    cn = coords.copy()
    if box =='rectangular':
        fxu = coords[:,0] > boxsize[0]
        fxl = coords[:,0] < 0
        fyu = coords[:,1] > boxsize[1]
        fyl = coords[:,1] < 0
        fzu = coords[:,2] > boxsize[2]
        fzl = coords[:,2] < 0
        cn[:,0] -= boxsize[0]*np.array(fxu,dtype=float)
        cn[:,0] += boxsize[0]*np.array(fxl,dtype=float)
        cn[:,1] -= boxsize[1]*np.array(fyu,dtype=float)
        cn[:,1] += boxsize[1]*np.array(fyl,dtype=float)
        cn[:,2] -= boxsize[2]*np.array(fzu,dtype=float)
        cn[:,2] += boxsize[2]*np.array(fzl,dtype=float)
    
    return cn 

@jit(nopython=True,fastmath=True)
def square_diff(x,c):
    return ((x-c)/c)**2
@jit(nopython=True,fastmath=True)
def norm2(r):
    x =0
    for i in range(r.shape[0]): 
        x += r[i]*r[i]
    return x**0.5
@jit(nopython=True,fastmath=True)
def norm2_axis1(r):
    return (r*r).sum(axis=1)**0.5


@jit(nopython=True,fastmath=True)
def calc_dist(r1,r2):
    r = r2 - r1
    d = np.sqrt(np.dot(r,r))
    return d

@jit(nopython=True,fastmath=True)
def calc_angle(r1,r2,r3):
    d1 = r1 -r2 ; d2 = r3-r2
    nd1 = np.sqrt(np.dot(d1,d1))
    nd2 = np.sqrt(np.dot(d2,d2))
    cos_th = np.dot(d1,d2)/(nd1*nd2)
    return np.arccos(cos_th)

@jit(nopython=True,fastmath=True)
def calc_dihedral(r1,r2,r3,r4):
    d1 = r2-r1
    d2 = r3-r2
    d3 = r4-r3
    c1 = np.cross(d1,d2)
    c2 = np.cross(d2,d3)
    n1 = c1/np.sqrt(np.dot(c1,c1))
    n2 = c2/np.sqrt(np.dot(c2,c2))
    m1= np.cross(n1,d2/np.sqrt(np.dot(d2,d2)))
    x= np.dot(n1,n2)
    y= np.dot(m1,n2)
    dihedral = np.arctan2(y, x)
    return dihedral



def rearrange_dict_keys(dictionary):
    '''
    Changes the order of the keys to access data
    Parameters
    ----------
    d : Dictionary of dictionaries with the same keys.
    Returns
    -------
    x : Dictionary with the second set of keys being now first.

    '''
    x = {k2 : {k1:None for k1 in dictionary} for k2 in dictionary[list(dictionary.keys())[0]]}
    for k1 in dictionary:
        for k2 in dictionary[k1]:
            x[k2][k1] = dictionary[k1][k2]
    return x
  
def check_occurances(a):
    x = set()
    for i in a:
        if i not in x:
            x.add(i)
        else:
            raise Exception('{} is more than once in the array'.format(i))
    return

class maps:
    charge_map = {'CD':-0.266,'C':0.154,'CE':0.164,
                  'hC':-0.01,'hCD':0.132,'hCE':-0.01
                      }
        
elements_mass = {'H' : 1.008,'He' : 4.003, 'Li' : 6.941, 'Be' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'Ne' : 20.180, 'Na' : 22.990, 'Mg' : 24.305,\
                 'Al' : 26.982, 'Si' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'Cl' : 35.453, 'Ar' : 39.948, 'K' : 39.098, 'Ca' : 40.078,\
                 'Sc' : 44.956, 'Ti' : 47.867, 'V' : 50.942, 'Cr' : 51.996,\
                 'Mn' : 54.938, 'Fe' : 55.845, 'Co' : 58.933, 'Ni' : 58.693,\
                 'Cu' : 63.546, 'Zn' : 65.38, 'Ga' : 69.723, 'Ge' : 72.631,\
                 'As' : 74.922, 'Se' : 78.971, 'Br' : 79.904, 'Kr' : 84.798,\
                 'Rb' : 84.468, 'Sr' : 87.62, 'Y' : 88.906, 'Zr' : 91.224,\
                 'Nb' : 92.906, 'Mo' : 95.95, 'Tc' : 98.907, 'Ru' : 101.07,\
                 'Rh' : 102.906, 'Pd' : 106.42, 'Ag' : 107.868, 'Cd' : 112.414,\
                 'In' : 114.818, 'Sn' : 118.711, 'Sb' : 121.760, 'Te' : 126.7,\
                 'I' : 126.904, 'Xe' : 131.294, 'Cs' : 132.905, 'Ba' : 137.328,\
                 'La' : 138.905, 'Ce' : 140.116, 'Pr' : 140.908, 'Nd' : 144.243,\
                 'Pm' : 144.913, 'Sm' : 150.36, 'Eu' : 151.964, 'Gd' : 157.25,\
                 'Tb' : 158.925, 'Dy': 162.500, 'Ho' : 164.930, 'Er' : 167.259,\
                 'Tm' : 168.934, 'Yb' : 173.055, 'Lu' : 174.967, 'Hf' : 178.49,\
                 'Ta' : 180.948, 'W' : 183.84, 'Re' : 186.207, 'Os' : 190.23,\
                 'Ir' : 192.217, 'Pt' : 195.085, 'Au' : 196.967, 'Hg' : 200.592,\
                 'Tl' : 204.383, 'Pb' : 207.2, 'Bi' : 208.980, 'Po' : 208.982,\
                 'At' : 209.987, 'Rn' : 222.081, 'Fr' : 223.020, 'Ra' : 226.025,\
                 'Ac' : 227.028, 'Th' : 232.038, 'Pa' : 231.036, 'U' : 238.029,\
                 'Np' : 237, 'Pu' : 244, 'Am' : 243, 'Cm' : 247, 'Bk' : 247,\
                 'Ct' : 251, 'Es' : 252, 'Fm' : 257, 'Md' : 258, 'No' : 259,\
                 'Lr' : 262, 'Rf' : 261, 'Db' : 262, 'Sg' : 266, 'Bh' : 264,\
                 'Hs' : 269, 'Mt' : 268, 'Ds' : 271, 'Rg' : 272, 'Cn' : 285,\
                 'Nh' : 284, 'Fl' : 289, 'Mc' : 288, 'Lv' : 292, 'Ts' : 294,\
                 'Og' : 294}


@jit(nopython=True,fastmath=True)
def numba_dipoles(pc,coords,segargs,dipoles):
    n = segargs.shape[0]
    for i in prange(n):
        cargs = segargs[i]
        dipoles[i] = np.sum(pc[cargs]*coords[cargs],axis=0)
    return
@jit(nopython=True,fastmath=True,parallel=True)
def numba_isin(x1,x2,f):
    for i in prange(x1.shape[0]):
        for x in x2:
            if x1[i] == x: 
                f[i] = True
    return
@jit(nopython=True,fastmath=True,parallel=True)
def numba_CM(coords,ids,mass,cm):
    for i in prange(ids.shape[0]):
        ji = ids[i]
        cm[i] = CM(coords[ji],mass[ji])
    return
