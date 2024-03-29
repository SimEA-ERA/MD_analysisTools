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
from scipy.integrate import simpson
from pytrr import GroTrrReader
import pytrr
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import logging
import coloredlogs
import matplotlib
import re
import lammpsreader 



class logs():
    '''
    A class for modifying the loggers
    '''
    def __init__(self):
        self.logger = self.get_logger()
        
    def get_logger(self):
    
        LOGGING_LEVEL = logging.DEBUG
        
        logger = logging.getLogger(__name__)
        logger.setLevel(LOGGING_LEVEL)
        logFormat = '%(asctime)s\n[ %(levelname)s ]\n[%(filename)s -> %(funcName)s() -> line %(lineno)s]\n%(message)s\n --------'
        formatter = logging.Formatter(logFormat)
        

        if not logger.hasHandlers():
            logfile_handler = logging.FileHandler('md_analysis.log',mode='w')
            logfile_handler.setFormatter(formatter)
            logger.addHandler(logfile_handler)
            self.log_file = logfile_handler
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
        return logger
    def __del__(self):
        self.logger.handlers.clear()
        self.log_file.close()

logobj = logs()        
logger = logobj.logger



class ass():
    '''
    The ASSISTANT class
    Contains everything that is needed to assist in the data analysis including
    1) functions for manipulating data in dictionaries
    2) The wrapper for using the same function and multyple trajectories on the Dynamic analysis
    3) printing and logger functions e.g. print_time, clear_logs 
    
    '''
    beebbeeb = True
    @staticmethod
    def try_beebbeeb():
        if ass.beebbeeb:
            try:
                import winsound
                winsound.Beep(500, 1000)
                import time
                time.sleep(0.5)
                winsound.Beep(500, 1000)
            except:
                pass
        return
    
    @staticmethod
    def change_key(d,keyold,keynew):
        try:
            value = d[keyold]
        except KeyError as e:
            raise Exception('{} This key "{}" does not belong to the dictionary'.format(e,keyold))
        d[keynew] = value
        del d[keyold]
        return 
    
    @staticmethod
    def dict_slice(d,i1,i2):
        return {k:v for i,(k,v) in enumerate(d.items()) if i1<=i<i2 }
    
    @staticmethod
    def numpy_keys(d):
        return np.array(list(d.keys()))
    
    @staticmethod
    def numpy_values(d):
        return np.array(list(d.values()))
    @staticmethod
    def common_keys(d1,d2):
        k1 = ass.numpy_keys(d1) ; k2 = ass.numpy_keys(d2)
        return np.intersect1d(k1,k2)
    @staticmethod
    def trunc_at(dold,dnew):
        
        new_key_first = list(dnew.keys())[1]
        dko = ass.numpy_keys(dold)
        if len(dold) ==0:
            return 0
        try:
            fa = dko< new_key_first
        except:
            return 0
        else:
            dl = dko[fa]
            itrunc = np.where(dl[-1] == dko)[0][0]
            return itrunc
    

    @staticmethod
    def is_dict_of_dicts(data):
        ks = list(data.keys())
        try:
            v0 = data[ks[0]]
            if type(v0) is dict:
                return True
            else:
                return False
        except IndexError:
            return False
        
    @staticmethod
    def wrap_multiple_trajectory(files,function,*fargs,**fkwargs):
        
        data_old = dict()
        mult_data = dict()
        
        for file in files:
            data = function(file,*fargs,**fkwargs)
            

            if ass.is_dict_of_dicts(data):
                data = ass.rearrange_dict_keys(data)
            
            trunc_at = ass.trunc_at(data_old,data)
            trunc_data_old = ass.dict_slice(data_old,0,trunc_at)
            
            mult_data.update(trunc_data_old)
            data_old = data
            logger.info('wrapped file {:s}'.format(file))
        mult_data.update(data)
        if ass.is_dict_of_dicts(mult_data):
            mult_data = ass.rearrange_dict_keys(mult_data)
        
        ass.clear_logs()
        
        return mult_data
    
    @staticmethod
    def print_time(tf,name,nf=None):
        s1 = ass.readable_time(tf)
        if nf is None:
            s2 =''
        else:
            s2 = ' Time/frame --> {:s}\n'.format( ass.readable_time(tf/nf))
        x = '-'*(len(name)+11)
        logger.info('Function "{:s}"\n{:s} Total time --> {:s}'.format(name,s2,s1))
    
    @staticmethod
    def print_stats( stats):
        print('ads chains  = {:4.4f} %'.format(stats['adschains_perc']*100))
        x = [stats[k] for k in stats if '_perc' in k and k.split('_')[0] in ['train','tail','loop','bridge'] ]
        tot = np.sum(x)
        for k in ['train','loop','tail','bridge']:
            print('{:s} = {:4.2f}%'.format(k,stats[k+'_perc']/tot*100))
        return
    
    @staticmethod
    def stay_True(dic):
        keys = list(dic.keys())
        stayTrue = {keys[0]:dic[keys[0]]}
        for i in range(1,len(dic)):
            stayTrue[keys[i]] = np.logical_and(stayTrue[keys[i-1]],dic[keys[i]])
        return stayTrue
    
    @staticmethod
    def become_False(dic):
        keys = list(dic.keys())
        bFalse = {keys[0]:dic[keys[0]]}
        for i in range(1,len(dic)):
            bFalse[keys[i]] = np.logical_and(bFalse[keys[0]],np.logical_not(dic[keys[i]]))
        return bFalse

    @staticmethod
    def iterable(arg):
        return (
            isinstance(arg, collections.Iterable) 
            and not isinstance(arg, six.string_types)
        )

    @staticmethod
    def readable_time(tf):
        hours = int(tf/3600)
        minutes = int((tf-3600*hours)/60)
        sec = tf-3600*hours - 60*minutes
        dec = sec - int(sec)
        sec = int(sec)
        return '{:d}h : {:d}\' : {:d}" : {:0.3f}"\''.format(hours,minutes,sec,dec)
    
    @staticmethod
    def rearrange_dict_keys(dictionary):
        '''
        Changes the order of the keys to access data
        Parameters
        ----------
        dictionary : Dictionary of dictionaries with the same keys.
        Returns
        -------
        x : Dictionary with the second set of keys being now first.
    
        '''
        x = {k2 : {k1:None for k1 in dictionary} for k2 in dictionary[list(dictionary.keys())[0]]}
        for k1 in dictionary:
            for k2 in dictionary[k1]:
                x[k2][k1] = dictionary[k1][k2]
        return x
    
    @staticmethod  
    def check_occurances(a):
        x = set()
        for i in a:
            if i not in x:
                x.add(i)
            else:
                raise Exception('{} is more than once in the array'.format(i))
        return     
    

@jit(nopython=True,fastmath=True,parallel=True)
def distance_kernel(d,coords,c):
    #Kernels for finding distances
    nd = coords.shape[1]
    for i in prange(d.shape[0]):
        d[i] = 0
        for j in range(nd):
            rel = coords[i][j]-c[j]
            d[i] += rel*rel
        d[i] = d[i]**0.5
    return 


@jit(nopython=True,fastmath=True)
def smaller_distance_kernel(d1,d2,c1,c2):
    #Kernel for finding the minimum distance between two coords
    for i in range(c1.shape[0]):
        distance_kernel(d2,c2,c1[i])
        d1[i] = 10000000000
        for j in range(d2.shape[0]):
            if d2[j]<d1[i]: 
                d1[i] = d2[j]
    return

@jit(nopython=True,fastmath=True)
def running_average(X,every=1):
    # curnel to find running average
    n = X.shape[0]
    xrun_mean = np.zeros(n)
    for j in range(0,len(X),every):
        y = X[:j+1]
        n = y.shape[0]
        xrun_mean[j] = np.sum(y)/n
    return xrun_mean

def moving_average(a, n=10) :
    #moving average kernel
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
    #block average kernel
    if a.shape[0]%n!=0:
        p = 1
    else:
        p=0
    bv = np.empty(int(a.shape[0]/n)+1,dtype=float)
    for i in range(bv.shape[0]):
        x = a[i*n:(i+1)*n]
        bv[i] = x.mean()
    return bv

def block_std(a, n=100) :
    #block standard diviation
    bstd = np.empty(int(a.shape[0]/n),dtype=float)
    for i in range(bn.shape[0]):
        x = a[i*n:(i+1)*n]
        bstd[i] = x.std()
    return bstd


class Energetic_Analysis():
    '''
    class to analyze gromacs energy files
    Currently needs improvement
    '''    
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
        if fname is not None:
            if path is not None:
                plt.savefig('{}\{}'.format(path,fname),bbox_inches='tight')
            else:
                plt.savefig('{}'.format(fname),bbox_inches='tight')
        plt.show()
        
class XPCS_Reader():
    def __init__(self,fname):
        self.relaxation_modes = []
        self.params = []
        self.params_std = []
        self.func = fitFuncs.freq
        with open( fname,'r') as f:
            lines = f.readlines()
            f.closed
            
        for i,line in enumerate(lines):
            l = line.strip() 
            if 'Background' in l:
                self.background= float(l.split(':')[1])
            if 'log Lagrange Multiplier' in l:
                self.reg = float(l.split(':')[1])
            if 'log(Upsilon)' in line:
                linum = i+2
        
        for i,line in enumerate(lines[linum:]):
            l = line.strip().split()
            if '----' in line: break
            self.relaxation_modes.append(float(l[0]))
            self.params.append(float(l[1]))
            self.params_std.append(float(l[2]))
        
        self.params = np.array(self.params)
        self.params_std= np.array(self.params_std)
        
        dlogtau = self.relaxation_modes[1]-self.relaxation_modes[0]
        w = self.params
        
        self.smoothness = np.sqrt (np.sum( (w[2:]-2*w[1:-1]+w[0:-2])**2)/dlogtau**4)
        self.relaxation_modes = 10**np.array(self.relaxation_modes)
        f = self.relaxation_modes
        self.bounds=(f[0],f[-1])
        return
    
    def fitted_curve(self,x):
        fc = self.func(x,self.bounds[0],self.bounds[1],self.params)
        try:
            fc += self.background
        except AttributeError:
            pass
        
        return fc
    
class plotter():
    def __init__(self):
        return
    
    class colors():
        qualitative = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
        diverging = ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf','#d9ef8b','#a6d96a','#66bd63','#1a9850']
        sequential = ['#fee0d2','#fc9272','#de2d26']
        safe = ['#1b9e77','#d95f02','#7570b3']
        semisafe = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
    class linestyles():
        lst_map = {
            'loosely dotted':      (0, (1, 3)),
            'dotted':               (0, (1, 1)),
            'densely dotted':       (0, (2, 1)),
            
            'loosely dashed':       (0, (5, 3)),
            'dashed':               (0, (4, 2)),
            'densely dashed':       (0, (3, 1)),

         'loosely dashdotted':   (0, (5, 3, 1, 3)),
         'dashdotted':           (0, (4, 2, 1, 2)),
         'densely dashdotted':    (0, (3, 1, 1, 1)),

         'dashdotdotted':         (0, (5, 2, 1, 2, 1, 2)),
         'loosely dashdotdotted': (0, (4, 3, 1,3, 1, 3)),
         'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
         }
        lst3 = [lst_map['densely dotted'],lst_map['loosely dashed'],
              lst_map['densely dashed']]
        lst4 = [lst_map['densely dotted'],lst_map['loosely dotted'],
              lst_map['densely dashed'],lst_map['loosely dashed']]
        lst6 = [lst_map['loosely dotted'],
                lst_map['dotted'],
                lst_map['dashed'],
                lst_map['loosely dashdotted'],
                lst_map['dashdotted'],
                lst_map['densely dashdotted']]
        lst7 = [lst_map['loosely dotted'],
                lst_map['dotted'],
                lst_map['dashed'],
                lst_map['loosely dashdotted'],
                lst_map['dashdotted'],
                lst_map['densely dashdotted'],
                lst_map['densely dotted']]
    @staticmethod
    def boldlabel(label):
        return r'$\mathbf{'+label+'}$'
    
    @staticmethod
    def relaxation_time_distributions(datadict,
                                      size=3.5,fname=None,title=None,
                                      cmap=None,xlim=(-6,6),
                                      units='ns',mode='tau'):
        
        if cmap is None:
            c = plotter.colors.semisafe
            cmap = { k : c[i] for i,k in enumerate(datadict.keys()) }
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')
        if title is not None: 
            plt.title(title)
        xticks = [10**x for x in range(xlim[0],xlim[1]+1) ]
        plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))
        plt.yticks(fontsize=2.5*size)
        if mode =='freq': 
            units = '{:s}^{:s}'.format(units,'{-1}')
            lab = 'freq'
        else:
            lab='\tau'
        plt.xlabel(r'${:s}$ / ${:s}$'.format(lab,units),fontsize=2*size)
        #plt.ylabel(r'$w$',fontsize=3*size)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for i,(k,f) in enumerate(datadict.items()):

            plt.plot(f.relaxation_modes,f.params,ls='--',lw=0.25*size,marker = 'o',
                markersize=size*1.2,markeredgewidth=0.15*size,fillstyle='none',color=cmap[k],label=k)
            #plt.plot(f.relaxation_modes,f.params,ls='--',label=k,color=cmap[k])
        plt.legend(frameon=False,fontsize=2.3*size)
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return 
    @staticmethod
    def compare_data(data1,data2,plot_args=[],**kwargs):
        if type(data1) is dict:
            if type(data2) is dict:
                multiple_data=True
            else:
                raise Exception('Either give both data as objects or as dicts')
            if len(data1) != len(data2):
                raise Exception('Data dicts must be of the same size')
        else:
            multiple_data = False

            
        if multiple_data == True:
            for i in range(len(data1)):
                if i>=len(plot_args):
                    plot_args.append(dict())
                
                plot_args[i] = {**kwargs,**plot_args[i]}
                
            for (k1,d1),(k2,d2),pargs in zip(data1.items(),data2.items(),plot_args):
                dc = {k1:d1,k2:d2}
                plotter.relaxation_time_distributions(dc,**pargs)
        else:
           dc = {'data1':data1,'data2':data2}
           plotter.relaxation_time_distributions(dc,**kwargs) 
        return                
    @staticmethod
    def ReadPlot_XPCSdistribution(files,**plot_kwargs):
        datadict = dict()
        for file in files:
            key = file.split('\\')[-1].split('.')[0]
            datadict[key] = XPCS_Reader(file)
        plotter.relaxation_time_distributions(datadict,**plot_kwargs)
        return datadict
    
    @staticmethod
    def write_xy_forXPCS(fname,x,y):
        data = np.zeros((x.shape[0],3),dtype=float)
        data[:,0] = x
        data[:,1] = y
        data[:,2]+= np.random.uniform(0,1e-9,x.shape[0])
        np.savetxt(fname,data)
        return
    
    @staticmethod
    def write_data_forXPCS(datadict,path='XPCS_data',cutf=None,midtime=None,num=100):
        
        if cutf is None:
            cutf ={k:10**10 for k in dataddict}
        for k,dy in datadict.items():
            
            fn = '{:s}\\xpcs_{:s}.txt'.format(path,k)
            
            x = ass.numpy_keys(dy)/1000
            y = ass.numpy_values(dy)
            t = x<=cutf[k]
            x = x[t]
            y = y[t]
            
            args = plotter.sample_logarithmically_array(x,midtime=midtime,num=num)
            xw = x[args]
            yw = y[args]
            plotter.write_xy_forXPCS(fn, xw,yw+1)
        return
        
    @staticmethod
    def sample_logarithmically_array(x,midtime=None,num=100,first_ten=True):
        
        
        if midtime is not None:
            if not ass.iterable(midtime):
                if type(midtime) is not float:
                    if type(midtime) is not int: 
                        raise Exception('midtime must be float,int or iterable of floats,ints')
                midtime = [float(midtime)]
            else:
                midtime = [float(m) for m in midtime]
            midtime.append(float(x[-1]))
        else:
            midtime = [ float(x[-1]) ]
    
        nm = len(midtime)

        args = np.array([0])
        if first_ten: 
            num-= nm*10
        num /= nm
        num = int(num)
        i=0
        for midt in midtime:
   
            if type(midt) is not float: 
                    raise Exception('midtime must be float or iterable of floats')
            mid = x[x<=midt].shape[0]
            if first_ten:
                fj = int(round(10**i,0))
                args = np.concatenate((args,[j for j in range(fj,fj+10)]+[fj-1]))
            try:
                lgsample = np.logspace(i,np.log10(mid),num=num).round(0) 
            
                args = np.concatenate( (args, np.array(lgsample,dtype=int)) )
            except ValueError as e:
                logger.warning('Excepted ValueError{}\nConsider increasing number of sumpling points'.format(e))
            i=np.log10(mid)
  
        return np.unique(args[args<x.shape[0]])

    @staticmethod
    def plotDynamics(datadict,style='points',locleg='best',ncolleg=1,
             fname =None,title=None,size=3.5,xlim=None,ylim=None,pmap=None,
             cmap = None,ylabel=None,units='ns',midtime=None,labmap=None,
             num=100,cutf=None,lstmap=None,write_plot_data=False,yscale=None,
             edgewidth=0.3,markersize=1.5):

        if labmap is None:
            labmap  = {k:k for k in datadict}
        if cmap is None:
            c = plotter.colors.semisafe
            cmap = { k : c[i] for i,k in enumerate(datadict.keys()) }
        if lstmap is None:
            lst = plotter.linestyles.lst7*10
            lstmap = {k:lst[i] for i,k in enumerate(datadict.keys())}
        elif type(lstmap) is str:
            lstmap = {k:lstmap for k in datadict}
        if ylabel is None:
            ylabel =r'$P_1(t)$'
        else:
            ylabel=ylabel
        if pmap is None:
            pmap = {k:'o' for k in datadict}
        if cutf is None:
            cutf ={k:10**10 for k in datadict}
        
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')
        if yscale is not None:
            plt.yscale(yscale)
        if xlim is None:
            plt.xticks(fontsize=2.5*size) 
        else:
            xticks = [10**x for x in range(xlim[0],xlim[1]+1) ]
            plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))
            locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if ylim is None:
            plt.yticks(fontsize=2.5*size)
        else:
            yticks = [10**y for y in range(ylim[0],ylim[1]+1)]
            plt.yticks(yticks,fontsize=min(2.5*size,2.5*size*8/len(yticks)))
            locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
            ax.yaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        plt.xlabel(r'$t ({})$'.format(units),fontsize=3*size)
        plt.ylabel(ylabel,fontsize=3*size)

        for i,(k,dy) in enumerate(datadict.items()):
            x = ass.numpy_keys(dy)/1000
            y = ass.numpy_values(dy)
            t = x<=cutf[k]
            x = x[t][1:]
            y = y[t][1:]
            if num == 'all':
                args = np.arange(0,x.shape[0],dtype='i')
            else:
                args = plotter.sample_logarithmically_array(x,midtime=midtime,num=num)
            
            if style == 'points':
                plt.plot(x[args],y[args],ls='none',
                marker = pmap[k],markeredgewidth=edgewidth*size,
                label=labmap[k], markersize=size*markersize,fillstyle='none',
                color=cmap[k])
            elif style=='lines':
                plt.plot(x[args],y[args],ls=lstmap[k],lw=size/1.5,
                label=labmap[k],color=cmap[k])
            else:
                raise NotImplemented('Implement here you own plotting style. Use elif statement')
        
        plt.legend(frameon=False,fontsize=2.0*size,loc=locleg,ncol=ncolleg)
        if fname is not None:plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return

    
    @staticmethod
    def fits(datadict,fitteddict,locleg='lower left',ncolleg=1,
             fname =None,title=None,size=3.5,xlim=(-6,6),
             cmap = None,ylabel=None,units='ns',midtime=None,
             num=100,cutf=None,write_plot_data=False):

        if write_plot_data:
            plotter.write_data_forXPCS(datadict,cutf=cutf,midtime=midtime)
        
        if cmap is None:
            c = c = plotter.colors.semisafe
            cmap = { k : c[i] for i,k in enumerate(datadict.keys()) }
        if ylabel is None:
            ylabel =r'$P_1(t)$'
        else:
            ylabel=ylabel
        if cutf is None:
            cutf ={k:10**10 for k in fitteddict}
        
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')
        
        xticks = [10**x for x in range(xlim[0],xlim[1]+1) ]
        plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))
        plt.yticks(fontsize=2.5*size)
        plt.xlabel(r'$t ({})$'.format(units),fontsize=3*size)
        plt.ylabel(ylabel,fontsize=3*size)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for i,(k,fitd) in enumerate(fitteddict.items()):
            dy = datadict[k]
            x = ass.numpy_keys(dy)/1000
            y = ass.numpy_values(dy)
            t = x<=cutf[k]
            x = x[t][2:]
            y = y[t][2:]
            
            args = plotter.sample_logarithmically_array(x,midtime=midtime,num=num)

 
            xf = np.logspace(xlim[0],xlim[1],base=10,num=10000)
            yee =fitd.fitted_curve(xf)
            plt.plot(xf,yee,ls ='-.',color=cmap[k],label ='fit {}'.format(k))
            
            plt.plot(x[args],y[args],ls='none',marker = 'o',markeredgewidth=0.2*size,
        markersize=size*1.2,fillstyle='none',color=cmap[k],label=k)
        plt.legend(frameon=False,fontsize=2.0*size,loc=locleg,ncol=ncolleg)
        if fname is not None:plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return

class dielectric_constants():
    def __init__(self,t,ft,de=1,omega=1.0):
        self.t = t
        self.ft = ft
        self.de=de
        self.omega=omega
        return
    def find_epsilon(self):
        
        def ep_epp(t,ft,de,o):
            
            I = simpson(ft*np.exp(-1j*o*t),x=t)
            eps = (1-1j*o*I)*de
            return eps
        if ass.iterable(self.omega):
            eps = [ ep_epp(self.t,self.ft,self.de,o) for o in self.omega]
        else:
            eps = ep_epp(self.t,self.ft,self.de,self.omega)
        return eps
    
    
class fitData():
    
    from scipy.optimize import dual_annealing,minimize
    
    def __init__(self,xdata,ydata,func,method='distribution',bounds=None,
                 search_grid=None,reg_bounds=None,minimum_res=1e-2,
                 nw=50,regd=1000,maxiter=200,**options):
        self.ydata = ydata
        self.xdata = xdata
        self.mode = func
        self.func = getattr(fitFuncs,func)
        
        if method == 'distribution':
            self.kernel = getattr(fitKernels,func)
        
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = (1e-7,1e7)
        if search_grid is not None:
            self.search_grid= search_grid
        else:
            self.search_grid= (12,4,4)
        if reg_bounds is not None:
            self.reg_bounds = reg_bounds
        else:
            self.reg_bounds = (1e0,1e8)
        self.minimum_res = minimum_res
        self.nw = nw
        self.maxiter=maxiter
        self.regd = regd
        
        if 'is_distribution' in options:
            self.is_distribution = options['is_distribution']
        else:
            self.is_distribution = True
            
        if 'zeroEdge_distribution' in options:
            self.zeroEdge_distribution = options['zeroEdge_distribution']
        else:
            self.zeroEdge_distribution = True
        
        if 'show_report' in options:
            self.show_report = options['show_report']
        else:
            self.show_report = True
        if 'init_method' in options:
            self.init_method = options['init_method']
        else:
            self.init_method = 'ones'
            
        if 'opt_method' in options:
            self.opt_method = options['opt_method']
        else:
            self.opt_method = 'SLSQP'
        return
            
    def search_best(self):
        self.crit = []
        self.smv = []
        self.drv = []
        self.regs = []
        
        irs = self.reg_bounds
        a = self.bounds[0]
        b = self.bounds[1]
        for numreg in self.search_grid:
            self.search_reg(a,b,numreg,irs)
            irs = self.last_regs[np.argsort(self.last_crit)[0:2]]
            irs.sort()
        rgbest = self.bestreg
        #self.maxiter = 10000
        self.exactFit(a,b,self.nw,rgbest)
        return
    
    def search_reg(self,a,b,numreg,irs=(1e0,1e8)):
        
        for attr in ['crit','smv','drv','regs']:
            try:
                dump = getattr(self,attr)
            except AttributeError:
                setattr(self,attr,[])
                
        lreg = np.logspace(np.log10(irs[0]),np.log10(irs[1]),base=10,num=numreg)
        #lreg = np.linspace(irs[0],irs[1],num=numreg)
        self.last_regs = lreg
        self.last_crit = []
        n = self.nw
        a = self.bounds[0]
        b = self.bounds[1]
        
        for reg in lreg:
            self.current_reg = reg
            self.exactFit(a,b,n,reg)
           # print('lr = {:.3e}'.format(lr))

            s = self.smoothness ; d = self.data_res
            crit = np.sqrt(s**2 + d**2)
            
            self.smv.append(s) ; self.drv.append(d)
            self.regs.append(reg) ;
            self.crit.append( crit)
            self.last_crit.append(crit)
            self.nsearches = len(self.regs)
            if self.show_report:
                self.report()
            #self.show_relaxation_modes(title='lr = {:.3e}'.format(lr))
        return
    
    @staticmethod
    def get_logtimes(a,b,n):
        tau = np.logspace(np.log10(a),np.log10(b),base=10,num=n)
        return tau
    
    def initial_params(self):
        if self.init_method =='uniform':
            pars = np.ones(self.nw)/self.nw
        elif self.init_method =='from_previous':
            try:
                pars = self.params
            except:
                pars =  np.ones(self.nw)/self.nw
        elif self.init_method =='ones':
            pars = np.ones(self.nw)
        return pars
    
    def exactFit(self,a,b,n,regs=1):
        
        tau = fitData.get_logtimes(a,b,n)
        dlogtau = np.log10(tau[1]/tau[0]) 
        x = self.xdata.copy()
        y = self.ydata.copy()
        ya = y.copy()
        
        A = self.kernel(x,tau)
        Da = A.copy()
        
        regd = self.regd
        if self.is_distribution:
            A = np.concatenate( (A, regd*np.ones((1,tau.shape[0])) ) )
            y = np.concatenate((y,regd*np.ones(1)))
        
        if self.zeroEdge_distribution:
            c1 = np.zeros((2,tau.shape[0]))
            c1[0][0] = regd
            c1[1][-1] = regd
            A = np.concatenate((A,c1))
            y = np.concatenate((y,np.zeros(2)))
        
        constraints = [ {'type':'eq',
                         'fun':lambda w:np.sum((np.dot(A,w)-y)**2)/y.shape[0]
                        } 
                      ]
        bounds = [(0,1) for i in range(n)]
        regcost = regs/dlogtau**4
        
        costf = lambda w: regcost*np.sum( (w[2:]-2*w[1:-1]+w[0:-2])**2)
        p0 = self.initial_params()   
        opt_res = minimize(costf,p0,bounds=bounds,method = self.opt_method,
                          constraints=constraints,
                          options = {'maxiter':self.maxiter,'disp':self.show_report})
        
        isd = 1-opt_res.x.sum() ; bl = opt_res.x[0] ; bu = opt_res.x[1]
        self.con_res = constraints[0]['fun'](opt_res.x)
        self.params = opt_res.x
        self.opt_res = opt_res
        self.prob_distr = {'isd':isd,'blow':bl,'bup':bu}
        self.relaxation_modes = tau
        self.smoothness = np.sqrt (costf(opt_res.x)/regs)
        self.data_res = np.sqrt(np.sum((np.dot(Da,opt_res.x)-ya)**2)/ya.shape[0] )
        return opt_res
    
    def report(self):
        for k in ['prob_distr','con_res','data_res','smoothness',
                  'current_reg','nsearches']:
            a = getattr(self,k)
            print('{:s} = {}'.format(k,a))
        return
    @property
    def bestreg(self):
        regs = np.array(self.regs)
        d = np.array(self.drv)
        print(regs.shape,d.shape)
        filt = d<self.minimum_res
        regs = regs[filt]
        crit = np.array(self.crit)[filt]
        print(regs.shape,crit.shape)
        try:
            regbest =  regs[np.argmin(crit)]
        except Exception as err:
            logger.debug('Something strange happend --> {}'.format(err))
            raise Exception(err)
        return regbest
      
    def print_reg(self):
        print('best reg = {:.6e}'.format(self.bestreg))
        return
    
    def show_Pareto_front(self,size=3.5,
                              title=None,color='magenta',fname=None):
        
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        d = np.array(self.drv)
        s = np.array(self.smv)

        plt.xscale('log')
        plt.yscale('log')
        plt.yticks(fontsize=2.5*size)
        plt.xlabel(r'smoothness',fontsize=3*size)
        plt.ylabel(r'data residual',fontsize=3*size)
        if title is None:
            plt.title('Pareto Front')
        else:
            plt.title(title)
        plt.plot(s,d,
                 ls='none',marker='o',color=color,markersize=1.7*size,fillstyle='none')
        
        pareto = []
        for i,(si,di) in enumerate(zip(s,d)):
            fs = si > s 
            fd = di > d
            f = np.logical_and(fs,fd)
            if f.any(): continue
            pareto.append(i)
        p = np.array(pareto,dtype=int)
        sp = s[p] ; dp =d[p]
        ser = sp.argsort()
        sp = sp[ser] ; dp = dp[ser]
        plt.ylim([0,dp.max()*1.05])
        plt.xlim([0,sp.max()*1.05])
        plt.plot(sp,dp,ls='--',color='k',lw=size/5)
        plt.plot([self.smoothness],[self.data_res],marker='o',color='red',markersize=1.7*size)
        
        plt.xticks(fontsize=2.5*size)
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return
    
    def show_relaxation_modes(self,size=3.5,units='ns',
                              title=None,color='red',fname=None):
        
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')
        
        plt.yticks(fontsize=2.5*size)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        plt.ylabel('w',fontsize=3*size)
        if self.mode=='freq':
            units = r'${:s}^{:s}$'.format(units,'{-1}')
            lab = r'$f$ / {:s}'.format(units)
        elif mode =='tau':
            lab = r'$\tau$ / {:s}'.format(units)
        plt.xlabel(lab,fontsize=3*size)
        if title is None:
            plt.title('Relaxation times distribution')
        else:
            plt.title(title)
        plt.plot(self.relaxation_modes,self.params,
                 ls='none',marker='o',color=color,markersize=1.7*size,fillstyle='none')
        xticks = [10**x for x in range(-10,20) if self.bounds[0]<=10**x<=self.bounds[1] ]
        plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return
    def fitted_data(self):
        return self.func(self.xdata,*self.params)
    
    def fitted_curve(self,x):
        
        return self.func(x,self.bounds[0],self.bounds[1],self.params)

class fitKernels():
    @staticmethod
    def freq(t,f):
        return np.exp(-np.outer(t,f))
    @staticmethod
    def tau(t,tau):
        return np.exp(-np.outer(t,(1/tau)))

class fitFuncs():
    @staticmethod
    def freq(t,a,b,w):
        frqs = fitData.get_logtimes(a,b,len(w))
        s=0
        for i,fr in enumerate(frqs):
            s+= w[i]*np.exp(-t*fr)
        return s
    @staticmethod
    def tau(t,a,b,w):
        taus = fitData.get_logtimes(a,b,len(w))
        s=0
        for i,ta in enumerate(taus):
            s+= w[i]*np.exp(-t*ta)
        return s
    
class Analytical_Expressions():
    @staticmethod
    def KWW_simple(t,beta,tww):
        #Kohlrausch–Williams–Watts
        #A is initial point shift
        #tc changes the shape of the fast motion(slow times)
        #beta changes the shape of the curve. 
        #very small beta makes the curve linear.
        #the larger the beta the sharpest the curve
        #all curves regardless beta pass through the same point
        phi = np.exp( -(t/tww )**beta )
        return phi


    @staticmethod
    def KWW(t,A,beta,tww):
        #Kohlrausch–Williams–Watts
        #A is initial point shift
        #tc changes the shape of the fast motion(slow times)
        #beta changes the shape of the curve. 
        #very small beta makes the curve linear.
        #the larger the beta the sharpest the curve
        #all curves regardless beta pass through the same point
        phi = A*np.exp( -(t/tww )**beta )
        return phi
    
    @staticmethod
    def KWW2(t,tw1,tw2,b1,b2,A):
        s = 0 
        tww = np.array([tw1,tw2])
        beta =np.array([b1,b2])
        for tw,b in zip(tww,beta):
            s+= Analytical_Expressions.KWW(t,A,b,tw)
        return s
    
    @staticmethod
    def expDecay_sum(t,t0v):
        s = np.zeros(t.shape[0])
        t0v = np.array(t0v)
        for i,t0 in enumerate(t0v):
            s+=Analytical_Expressions.expDecay_simple(t,t0)
        return s/t0v.shape[0]
    @staticmethod
    def expDecay_simple(t,t0):
        phi =  np.exp(-t/t0)
        return phi
    @staticmethod
    def expDecay(t,A,t0):
        #A paramater shifts the end point of the curve up to minus A, see exersize 2
        phi = 1+A*( np.exp(-t/t0) - 1 )
        return phi
    @staticmethod
    def expDecay2(t,A,t0):
        #A parameter shifts the starting point to A
        phi = A*np.exp(-t/t0)
        return phi
    @staticmethod
    def expDecay3(t,A,t0):
        #A parameter shifts and the end point to point to -A
        phi = A*(np.exp(-t/t0)-1)
        return phi  
    @staticmethod
    def expDecay4(t,As,Ae,t0):
        #As shifts initial point to As
        #Ae shifts the whole curve by Ae
        phi = As*np.exp(-t/t0)-Ae
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
    '''
    This class is used exclucively to add hydrogens on correct locations
    and be able to compute all atom properties like dipole momement vectors (dielectric data)
    Currently works well for PB. It is NOT tested for anything else. 
    It needs attention to the geometry calculations and should be used
    with the add_atoms class.
    '''
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
    '''
    This class is used to add atoms to the system
    Currently works well to add hydrogens to PB
    It needs modification to achieve generality
    '''
    def __init__(self):
        pass
    
    hydrogen_map = {'CD':[1,0.11,116,(1,),'_cis'],
                    'C':[2,0.11,109.47,(1,2),''],
                    'CE':[3,0.109,109.47,(1,2,3),'_endgroup']}
    
    @staticmethod
    def add_ghost_atoms(self,system2,gconnectivity=dict()):
        t0 = perf_counter()
        for s in ['at_ids','at_types','mol_names','mol_ids']:
            a1 = getattr( system2, s )
            setattr(self,'_'.join(('ghost',s)),a1)
        for frame in system2.timeframes:
            c1 = system2.timeframes[frame]['coords']
            self.timeframes[frame]['_'.join(('ghost','coords'))] = c1
        self.ghost_connectivity = gconnectivity
        self.mass_map.update(system2.mass_map)
        return
            
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
        ass.print_time(perf_counter()-t0,
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
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
    
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
    def set_ghost_coords(self,info):
       
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
    '''
    Depending on the confinemnt type one of these functions
    will be called.
    These class functions are used to calculate 
    the minimum image distance
    '''
    @staticmethod
    def zdir(box):
        return [box[2],0,-box[2]]
    @staticmethod
    def ydir(box):
        return [box[1],0,-box[1]]
    @staticmethod
    def xdir(box):
        return [box[0],0,-box[0]]
    @staticmethod
    def minimum_distance(box):
        zd = Box_Additions.zdir(box)
        yd = Box_Additions.ydir(box)
        xd = Box_Additions.xdir(box)
        lst_L = [np.array([x,y,z]) for x in xd for y in yd for z in zd]
        return lst_L
    @staticmethod
    def spherical_particle(box):
        zd = Box_Additions.zdir(box)
        yd = Box_Additions.ydir(box)
        xd = Box_Additions.xdir(box)
        lst_L = np.array([np.array([x,y,z]) for x in xd for y in yd for z in zd])
        return lst_L
    @staticmethod
    def zcylindrical(box):
        return [0]
        
class Distance_Functions():
    '''
    Depending on the confinemnt type one of these functions
    will be called.
    These class functions are used to calculate 
    the Distance between coords and a center position (usually particle center of mass is passed)
    '''
    @staticmethod
    def zdir(self,coords,zc):
        return np.abs(coords[:,2]-zc)
    @staticmethod
    def ydir(self,coords,yc):
        return np.abs(coords[:,1]-yc)
    @staticmethod
    def xdir(self,coords,xc):
        return np.abs(coords[:,0]-xc)
    

    
    @staticmethod
    def zdir__2side(self,coords,zc):
        return coords[:,2]-zc
    @staticmethod
    def ydir__2side(self,coords,yc):
        return coords[:,1]-yc
    @staticmethod
    def xdir__2side(self,coords,xc):
        return coords[:,0]-xc
    
    @staticmethod
    def spherical_particle(self,coords,c):
        d = np.zeros(coords.shape[0],dtype=float)
        distance_kernel(d,coords,c)
        #r = coords -c
        #d = np.sqrt(np.sum(r*r,axis=1))
        return d
    
    @staticmethod
    def minimum_distance(self,coords1,coords2):
        d1 = np.empty(coords1.shape[0])
        d2 = np.empty(coords2.shape[0])
        smaller_distance_kernel(d1,d2,coords1,coords2)
        return d1

    @staticmethod
    def zcylindrical(self,coords,c):
         #d = Distance_Functions.distance(coords[:,0:2],c[0:2])
         r = coords[:,0:2]-c[0:2]
         d = np.sqrt(np.sum(r*r,axis=1))
         #r = coords[:,0:2]-c[0:2] ; d = np.sqrt(np.sum(r*r,axis=1))
         d = self.global_cylindrical_radius-d
         #logger.debug('{:d} and {:d}'.format(d.shape[0],c.shape[0]))
         return d
     
class bin_Volume_Functions():
    '''
    Depending on the confinemnt type one of these functions
    will be called.
    These class functions are used to calculate 
    the volume of each bin when needed (e.g. for density profile calculations) 
    '''
    @staticmethod
    def zdir(self,bin_low,bin_up):
        box = self.get_box(self.current_frame)
        binl = bin_up-bin_low
        return 2*box[0]*box[1]*binl
    
    @staticmethod
    def ydir(self,bin_low,bin_up):
        box = self.get_box(self.current_frame)
        binl = bin_up-bin_low
        return 2*box[0]*box[2]*binl
    
    @staticmethod
    def xdir(self,bin_low,bin_up):
        box = self.get_box(self.current_frame)
        binl = bin_up-bin_low
        return 2*box[1]*box[2]*binl

    @staticmethod
    def zcylindrical(self,bin_low,bin_up):
        box = self.get_box(self.current_frame)
        binl = bin_up - bin_low
        rm = self.global_cylindrical_radius -(bin_low + 0.5*binl)
        if rm<0: return 1e16 # negative has no meaning therefore we give a large number to devide with it and make density zero
        return 2*np.pi*rm*binl*box[2]
    
    @staticmethod
    def spherical_particle(self,bin_low,bin_up):
        v = 4*np.pi*(bin_up**3-bin_low**3)/3
        return  v
    
    

class Center_Functions():
    '''
    Depending on the confinemnt type one of these functions
    will be called.
    returns the nessary coordinates of the center. 
    For example for confinement of flat surfaces on the z-direction (zdir)
    it returns just the z coordinate.
    For cylindrical on z returns the xy directions
    This is necessary to broadcast correctly and fast the relevant distances within the algorithm
    '''
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
    def spherical_particle(c):
        return c
    @staticmethod
    def zcylindrical(c):
        return c
    
class unit_vector_Functions():
    '''
    Depending on the confinemnt type one of these functions
    will be called.
    A unit vector for computing bond order is defined
    '''
    @staticmethod
    def zdir(self,r,c):
        uv = np.zeros((r.shape[0],3))
        uv[:,2] = 1
        return uv
    @staticmethod
    def ydir(self,r,c):
        uv = np.zeros((r.shape[0],3))
        uv[:,1] = 1
        return uv
    
    @staticmethod
    def xdir(self,r,c):
        uv = np.zeros((r.shape[0],3))
        uv[:,2] = 1
        return uv
    
    @staticmethod
    @jit(nopython=True,fastmath =True,parallel=True)
    def spherical_particle_inner(cperiodic,r,c,box_add):
        for i in prange(cperiodic.shape[0]):
            dist_old= 1e16    
            for j,L in enumerate(box_add):
                rr = r[i] - (c+L)
                dist = np.sum(rr*rr)
                if dist<dist_old:
                    jmin= j
                    dist_old = dist
            cperiodic[i] = c +box_add[jmin]
        return
    @staticmethod
    def spherical_particle(self,r,c):
        box = self.get_box(self.current_frame)
        box_add = Box_Additions.spherical_particle(box)
        cperiodic = np.empty(r.shape)
        unit_vector_Functions.spherical_particle_inner(cperiodic,r,c,box_add)
        uv = r-cperiodic
        return uv
    @staticmethod
    def zcylindrical(self,r,c):
        uv = np.ones((r.shape[0],3))
        uv[:,2] = 0
        return uv



    
class Analysis:
    '''
    The mother class. It's for simple polymer systems not confined ones
    '''
    def __init__(self,
                 trajectory_file, # gro/trr for now
                 connectivity_file, #itp
                 topol_file = None,
                 memory_demanding=False,
                 types_from_itp=True):
        t0 = perf_counter()
        self.trajectory_file = trajectory_file
        self.connectivity_file = connectivity_file
        self.memory_demanding = memory_demanding
        self.types_from_itp = types_from_itp
        '''
        trajectory_file: the file to analyze
        'connectivity_file': itp or list of itps. From the bonds it finds angles and dihedrals
        'gro_file': one frame gro file to read the topology (can be extended to other formats). It reads molecule name,molecule id, atom type
        memory_demanding: If True each time we loop over frames, these are readen from the disk and are not stored in memory
        types_from_itp: if different types excist in itp and gro then the itp ones are kept
        '''
        
        
        if '.gro' == trajectory_file[-4:]:
            self.traj_file_type = 'gro'
            self.topol_file = trajectory_file
            self.read_by_frame = self.read_gro_by_frame # function
            self.traj_opener = open
            self.traj_opener_args = (self.trajectory_file,)
        elif '.trr' == trajectory_file[-4:]:
            self.traj_file_type ='trr'
            self.read_by_frame =  self.read_trr_by_frame # function
            self.traj_opener = GroTrrReader
            self.traj_opener_args = (self.trajectory_file,)
            if '.gro' == topol_file[-4:]: 
                self.topol_file = topol_file
            else:
                s = 'Your  trajectory file is {:s} while your topol file is {:s}.\n \
                Give the right format for the topol_file. i.e. ".gro"\n \
                Only One frame is needed to get the \
                topology'.format(trajectory_file,gro_file)
                raise Exception(s)
        elif '.lammpstrj' == self.trajectory_file[-10:]:
            self.traj_file_type ='lammpstrj'
            self.read_by_frame = self.read_lammpstrj_by_frame
            self.traj_opener = lammpsreader.LammpsTrajReader
            self.traj_opener_args = (self.trajectory_file,)
            if topol_file is None:
               topol_file = self.connectivity_file
            self.topol_file = topol_file
        else:
            raise NotImplementedError('Trajectory file format ".{:s}" is not yet Implemented'.format(trajectory_file.split('.')[-1]))
        self.read_topology()
        
        self.analysis_initialization()
        
        self.timeframes = dict() # we will store the coordinates,box,step and time here
        tf = perf_counter()-t0
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return 
    
    def analysis_initialization(self):
        '''
        Finds some essential information for the system and stores it in self
        like angles,dihedrals
        Also makes some data manipulation that might be used later
        '''
        t0 = perf_counter()
        ## We want masses into numpy array
        self.find_masses()
        
        #Now we want to find the connectivity,angles and dihedrals
        
        self.find_neibs()
        self.find_angles()
        self.find_dihedrals()
        #Find the ids (numpy compatible) of each type and store them
        self.keysTotype('connectivity')
        self.keysTotype('angles')
        self.keysTotype('dihedrals')
        
        self.find_unique_bond_types()
        self.find_unique_angle_types()
        self.find_unique_dihedral_types()
        self.dict_to_sorted_numpy('connectivity') #necessary to unwrap the coords efficiently
        
        self.find_args_per_residue(np.ones(self.mol_ids.shape[0],dtype=bool),'molecule_args')
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
            
        return
    
    def update_connectivity(self,bid):
        bid,bt = self.sorted_id_and_type(bid)
        self.connectivity[bid] = bt
        return
    
    @property
    def nframes(self):
        return len(self.timeframes)
    
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
        setattr(self,'sorted_'+attr_name+'_keys',x)
        
        tf = perf_counter() - t0 
        #ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return

    def keysTotype(self,attr_name):
        dictionary = getattr(self,attr_name)
        if type(dictionary) is not dict:
            raise TypeError('This function is for working with dictionaries')
        types = self.unique_values(dictionary.values())
        temp_ids = {i:[] for i in types}
        for k,v in dictionary.items():
            temp_ids[v].append(np.array(k))
        ids = {v : np.array(temp_ids[v]) for v in types}
        setattr(self,attr_name+'_per_type',ids)
        return 
    
    def find_topology_from_itp(self):
        #t0 = perf_counter()

        if ass.iterable(self.connectivity_file):
            for cf in self.connectivity_file:
                if '.itp' in cf:
                    self.read_atoms_and_connectivity_from_itp(cf)
                    itp =True           
                else:   
                    raise NotImplementedError('Non itp files are not yet implemented')
                
        else:
            if '.itp' == self.connectivity_file[-4:]:
                self.read_atoms_and_connectivity_from_itp(self.connectivity_file)
                itp =True
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

        
        #tf = perf_counter() - t0
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)
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
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
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
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
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
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return frame_min
    
    def sorted_id_and_type(self,a_id):
        t = [self.at_types[i] for i in a_id]
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
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)
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
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)
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
            #ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
            
            return np.array(at_ids),np.array(at_types),np.array(res_num),np.array(res_name),atoms,cngr,charge,mass
 
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
    
        
        at_ids,at_types,res_num,res_name,atoms,cngr,charge,mass \
        = read_atoms_from_itp(file)
        
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
        bonds = read_connectivity_from_itp(file)
        
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
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return 
    

    def read_topology(self):
        if '.gro' == self.topol_file[-4:]:
            self.read_gro_topol()  # reads from gro file
            self.find_topology_from_itp() # reads your itp files to get the connectivity
        elif '.ltop' == self.topol_file[-5:]:
            self.read_lammps_topol()
        return
    
    def read_lammps_topol(self):
        t0 = perf_counter()
        with open(self.topol_file,'r') as f:
            lines = f.readlines()
            f.closed
        
        def get_value(lines,valuename,dtype=int):
            for line in lines:
                if valuename in line:
                    return dtype(line.split()[0])
        
        def get_line_of_header(lines,header):
            for i,line in enumerate(lines):
                if header in line:
                    return i
        values = ['atoms','bonds','angles','dihedrals','impropers',
                  'atom types', 'bond types', 'angle types', 
                  'dihedral types','improper types']
        headers = ['Masses','Atoms','Bonds','Angles','Dihedrals']
        
        numbers = {v:get_value(lines,v) for v in values}
        
        header_lines = {hl:get_line_of_header(lines,hl) for hl in headers}
        
        natoms = numbers['atoms']
        mol_ids = np.empty(natoms,dtype=int)
        mol_nms = np.empty(natoms,dtype=object)
        at_tys  = np.empty(natoms,dtype=object)
        at_ids  = np.empty(natoms,dtype=int)
        hla = header_lines['Atoms']
        atom_lines = lines[hla+2 : hla+2+natoms]
        ncols = len(atom_lines[0].split())
        self.charge_map = dict()
        for i,line in enumerate(atom_lines):
            l = line.strip().split()
            mol_ids[i] = int(l[1])
            mol_nms[i] = l[1]
            at_tys[i] = l[2]
            at_ids[i] = int(l[0])
            if ncols==4 or ncols ==7 or ncols==10:
                self.charge_map[l[2]] = float(l[3])
            
        sort_ids = at_ids.argsort()
        mol_ids = mol_ids[sort_ids]
        mol_nms = mol_nms[sort_ids]
        at_tys = at_tys[sort_ids]
        at_ids = at_ids[sort_ids]
        starts_from = at_ids.min()
        for i in range(1,natoms):
            if at_ids[i-1]+1 != at_ids[i]:
                raise Exception('There are missing atoms')
        at_ids -= starts_from
        
        self.mol_ids = mol_ids
        self.mol_names = mol_nms
        self.at_types = at_tys
        self.at_ids = at_ids
        
        self.find_locGlob()
        
        self.connectivity= dict()

        hlb = header_lines['Bonds'] 
        nbonds = numbers['bonds']
        bond_lines = lines[hlb+2:hlb + nbonds+2]
        for  i,line in enumerate(bond_lines):  
            b = line.strip().split() 
            id0 = int(b[-2]) - starts_from
            id1 = int(b[-1]) - starts_from
            conn_id,c_type =  self.sorted_id_and_type((id0,id1))
            if conn_id in self.connectivity:
                logger.warning('{} is already in connectivity '.format(conn_id))
            self.connectivity[conn_id] = c_type
        
        self.mass_map = dict()
        hlm = header_lines['Masses']
        
        for line in lines[hlm+2:+hlm+2+numbers['atom types']]:
            l = line.strip().split()
            self.mass_map[l[0]] = float(l[1])
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return 
    
    def read_gro_topol(self):
        with open(self.topol_file,'r') as f:
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
                at_ids[i] = i
            f.close()
        

        self.mol_ids = mol_ids
        self.mol_names = mol_nms
        self.at_types = at_tys
        self.at_ids = at_ids
        
        self.find_locGlob()
 
        return 
    
    def find_locGlob(self):
        loc_id_to_glob = dict() ; glob_id_to_loc = dict()
        for j in np.unique(self.mol_ids):
            loc_id_to_glob[j] = dict()
            glob_id_to_loc[j] = dict()
            filt = self.mol_ids== j
            res_nm = np.unique(self.mol_names[filt])
            if res_nm.shape !=(1,):
                raise ValueError('many names for a residue, res_id = {:d}'.format(j))
            else:
                res_nm = res_nm[0]
            g_at_id = self.at_ids[filt]
            
            for i,g in enumerate(g_at_id):
                loc_id = i
                loc_id_to_glob[j][loc_id] = g
                glob_id_to_loc[j][g] = loc_id
        
        self.loc_id_to_glob = loc_id_to_glob
        self.glob_id_to_loc = glob_id_to_loc 
        return 
    def charge_from_maps(self):
        '''
        Updates a dictionary which contains charges for elements from the maps class
        '''
        self.charge_map.update(maps.charge_map)
        return
    
    @property
    def natoms(self):
        return self.at_ids.shape[0]
    @property
    def ndihedrals(self):
        return len(self.dihedrals)
    @property
    def nbonds(self):
        return len(self.connectivity)
    @property
    def nangles(self):
        return len(self.angles)
    @staticmethod
    def unique_values(iterable):
        try:
            iter(iterable)
        except:
            raise Exception('Give an ass.iterable variable')
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
    
    def find_masses(self):
        t0 = perf_counter()
        mass = np.empty(self.natoms,dtype=float)
        for i in range(self.natoms):
            mass[i] = self.mass_map[self.at_types[i]]
        self.atom_mass = mass
        tf = perf_counter() -t0
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)
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
        except EOFError:
            raise EOFError
        self.timeframes[frame] = header
        self.timeframes[frame]['boxsize'] = np.diag(data['box'])
        self.timeframes[frame]['coords'] = data['x']
        return True
   
    def read_lammpstrj_by_frame(self,reader,frame):
        conf = reader.readNextStep()
        if conf is None: 
            return False
        if not reader.isSorted(): reader.sort()                
        uxs = conf['xu']
        uys = conf['yu']
        uzs = conf['zu']
        # allocate 
        natoms = uxs.shape[0]
        coords = np.empty((natoms,3))
        coords[:,0] = uxs ; coords[:,1] = uys ; coords[:,2] = uzs
        cbox = conf['box_bounds']
        tricl = cbox[:,2] != 0.0
        if tricl.any():
            raise NotImplementedError('Triclinic boxes are not implemented')
        offset = cbox[:,0]
        boxsize = cbox[:,1]-cbox[:,0]
        coords -= offset
        #make nm
        boxsize/=10
        coords/=10
        try:
            dt = self.kwargs['dt']
        except KeyError:
            dt =1e-6
        self.timeframes[frame] = {'conf':conf,'time':conf['step_no']*dt,'step':conf['step_no'],
                                  'boxsize':boxsize,'coords':coords}
        return True

    def read_from_disk_or_mem(self,ofile,frame):
        def exceptEOF():
            try:
                ret = self.read_by_frame(ofile, frame)
            except EOFError:
                return False
            else:
                return ret
            
        if self.memory_demanding:
            return exceptEOF()
        elif frame in self.timeframes.keys():
            self.is_the_frame_read =True
            return True
        else:
            try:
                if self.is_the_frame_read:
                    return False
            except AttributeError:
               return exceptEOF()

    def read_lammpstrj_file(self):
        t0 = perf_counter()
        with lammpsreader.LammpsTrajReader(self.trajectory_file) as ofile:
            nframes = 0
            while( self.read_lammpstrj_by_frame(ofile, nframes)):
                if self.memory_demanding:
                    del self.timeframes[nframes]
                nframes += 1
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return
    
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
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
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
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return 
    
    def read_file(self):    
        if   self.traj_file_type == 'gro':
            self.read_gro_file()
        elif self.traj_file_type == 'trr': 
            self.read_trr_file()
        elif self.traj_file_type =='lammpstrj':
            self.read_lammpstrj_file()
        return 
    
    def write_gro_file(self,fname=None,option='',frames=None,step=None):
        t0 = perf_counter()
        options = ['','unwrap','transmiddle']
        if option not in options:
            raise ValueError('Available options are : {:s}'.format(', '.join(options)))
        
        if fname is None:
            fname = 'Analyisis_written.gro'
        with open(fname,'w') as ofile:
            for frame,d in self.timeframes.items():
                if frames is not None:
                    if  frame <frames[0] or frame>frames[1]:
                        continue
                if step is not None:
                    if frame%step  !=0: 
                        continue
                if option == 'unwrap':
                    coords = self.get_whole_coords(frame)
                elif option =='transmiddle':
                    coords = self.translate_particle_in_box_middle(self.get_coords(frame),
                                                                   self.get_box(frame))
                elif option=='':
                
                    coords = self.get_coords(frame)
                else:
                    raise NotImplementedError('option "{:}" Not implemented'.format(option))
                self.write_gro_by_frame(ofile,
                                        coords, d['boxsize'],
                                        time = d['time'], 
                                        step =d['step'])
            ofile.close()
        tf = perf_counter() -t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
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
    
    def renumber_residues(self,start_from=1):
        mol_ids = np.empty(self.natoms,dtype=int)        
        counter =start_from
        mol_ids[0] = start_from
        
        for i in range(1,self.natoms):
            mid0 = self.mol_ids[i-1]
            mid1 = self.mol_ids[i]
            if mid1!=mid0:
                counter+=1
            mol_ids[i] = counter
        self.mol_ids=mol_ids
    
        return
    
    def renumber_ids(self,start_from=0):
        at_ids = np.arange(0,self.natoms,1,dtype=int)
        at_ids+=start_from
        self.at_ids = at_ids
        return
    def filter_the_system(self,filt):
        self.at_ids = self.at_ids[filt]
        self.at_types = self.at_types[filt]
        self.mol_ids = self.mol_ids[filt]
        self.mol_names = self.mol_names[filt]
        for frame in self.timeframes:
            self.timeframes[frame]['coords'] = self.get_coords(frame)[filt]
        self.renumber_ids()
        self.renumber_residues()
        return
    def remove_residues(self,crit,frame=0,reinit=True):
        coords = self.get_coords(frame)
        filt = crit(coords)
        res_cut = self.mol_ids[filt]
        filt = ~np.isin(self.mol_ids,res_cut)
        self.at_ids = self.at_ids[filt]
        self.at_types = self.at_types[filt]
        self.mol_ids = self.mol_ids[filt]
        
        self.mol_names = self.mol_names[filt]
        for frame in self.timeframes:
            self.timeframes[frame]['coords'] = self.get_coords(frame)[filt]
        
        self.renumber_ids()
        self.renumber_residues()
        
        if reinit:
            self.analysis_initialization()
        return
    def write_residues(self,res,fname='selected_residues.gro',
                       frames=(0,0),box=None,boxoff=0.4):
        with open(fname,'w') as ofile:
            fres = np.isin(self.mol_ids, res)
            
            for frame in self.timeframes:
                if frames[0] <= frame <= frames[1]:

                    coords = self.get_coords(frame) [fres]
                    coords -= coords.min(axis=0)
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
                        box = coords.max(axis=0) - coords.min(axis=0) + boxoff
                    ofile.write('%f  %f  %f\n' % (box[0],box[1],box[2]))  
            
            ofile.closed
        return
    
    @property
    def nmolecules(self):
        return np.unique(self.mol_ids).shape[0]
    
    def find_args_per_residue(self,filt,attr_name):
        args = dict()
        for j in np.unique(self.mol_ids[filt]):
            x = np.where(self.mol_ids==j)[0]
            args[j] = x
        setattr(self,attr_name,args)
        setattr(self,'N'+attr_name, len(args))
        return 
    

    
    def multiply_periodic(self,multiplicity):
        if len(multiplicity) !=3:
            raise Exception('give the multyplicity in the form (times x, times y, times z)')
        for d,mult in enumerate(multiplicity):
            if mult ==0: continue
        #multiplicity= np.array([t for t in multiplicity])
            natoms = self.natoms
            nmols = self.nmolecules
            totm = mult+1
        
            if totm<2:
                raise Exception('multiplicity {} is not valid'.format(multiplicity))
            #new topology
            shape = natoms*totm
            # allocate array
            at_ids = np.empty(shape,dtype=int)
            mol_ids = np.empty(shape,dtype=int)
            at_types = np.empty(shape,dtype=object)
            mol_names = np.empty(shape,dtype=object)
            for m in range(0,mult+1):
                na = m*natoms
                mm = m*nmols
                for i in range(natoms):
                    idx = i+na
                    at_ids[idx] = idx
                    at_types[idx] =self.at_types[idx%natoms]
                    mol_ids[idx] = self.mol_ids[idx%natoms]+mm
                    mol_names[idx] = self.mol_names[idx%natoms]
            self.at_ids = at_ids
            self.at_types = at_types
            self.mol_ids = mol_ids
            self.mol_names = mol_names
        
            #allocate coords
            
            for frame in self.timeframes:
                coords = np.empty((shape,3))
                c = self.get_coords(frame)
                box = self.get_box(frame)
            
                idx = 0
                coords[idx:idx+natoms] = c.copy()
                for j in range(1,mult+1):
                    L = box[d]*j
                    idx=j*natoms
                    coords[idx:idx+natoms] = c.copy()
                    coords[idx:idx+natoms,d]+=L
            
                self.timeframes[frame]['coords'] = coords
                self.timeframes[frame]['boxsize'][d]*=mult+1
        
        return
    def unwrap_coords(self,coords,box):   
        '''
        Do not trust this function. Works only for linear polymers
        Parameters
        ----------
        coords : 
        box : 

        Returns
        -------
        unc : Unrwap coordinates.

        '''
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
        '''
        It's a Wrapper for looping over trajectories
        It takes a "core" function and it's arguments
        Then it loops over all trajectory calling the same
        function with it's arguments

        Parameters
        ----------
        fun : Function
        args : tuple of function arguments

        Returns
        -------
        nframes : the number of frames it had looped
        '''
        
        funtocall = getattr(coreFunctions,fun)
        if len(self.timeframes) == 0 or self.memory_demanding:
            with self.traj_opener(*self.traj_opener_args) as ofile:
                nframes=0
                while(self.read_from_disk_or_mem(ofile,nframes)):
                    self.current_frame = nframes
                    funtocall(self,*args)      
                    if self.memory_demanding:
                        del self.timeframes[nframes]
                    nframes+=1
        else:
            nframes = self.loop_timeframes(funtocall,args)
        del self.current_frame
        return nframes
    
    def loop_timeframes(self,funtocall,args):
        for frame in self.timeframes:
            self.current_frame = frame
            funtocall(self,*args)
        nframes = len(self.timeframes)
        return nframes
    
    @property
    def first_frame(self):
        if not self.memory_demanding:
            return list(self.timeframes.keys())[0]
        else:
            return 0
    
    def cut_timeframes(self,num_start=None,num_end=None):
        '''
        Used to cut the trajectory

        Parameters
        ----------
        num_start : int, frame the trajectory starts
        num_end :  int, frame the trajectory ends

        Returns
        -------
        None.

        '''
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
        new_dict = ass.dict_slice(self.timeframes,i1,i2)
        if len(new_dict) ==0:
            raise Exception('Oh dear you have cut all your timeframes from memory')
            
        self.timeframes = new_dict
        return 
    
    def calc_atomic_coordination(self,maxdist,type1,type2):
        
        t0 = perf_counter()
        def find_args_of_type(types):
            if not ass.iterable(types):
                types = [types]
            ids = np.array([],dtype=int)
            for ty in types:
                a1 = np.where([self.at_types==ty])[1]
                ids = np.concatenate((ids,a1))
            ids = np.unique(ids)
            return ids
        
        args1 = find_args_of_type(type1)
        args2 = find_args_of_type(type2)
        
        
        coordination = np.zeros(args1.shape[0])
        args = (maxdist,args1,args2,coordination)
        
        nframes = self.loop_trajectory('atomic_coordination',args)
        
        coordination/=nframes
        def tkey(ty):
            if ass.iterable(ty):
                k = '_'.join(ty)
            else:
                k = ty
            return k
        
        key ='-'.join([tkey(type1),tkey(type2)])
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return {key:coordination}
    
    def calc_pair_distribution(self,binl,dmax,type1=None,type2=None,
                               density='',normalize=False,**kwargs):
        '''
        Used to calculate pair distribution functions between two atom types
        Could be the same type. If one type is None then the distribution is 
        between the the other type and all atom types. If both is none then
        the distribution is between all atoms.

        Parameters
        ----------
        binl : FLOAT
            Binning legth.
        dmax : float
            maximum distance to calculate.
        type1 : string 
            . The default is None.
        type2 : string
            . The default is None.
        density : string
            DESCRIPTION. The default is ''.
            'number' is to return number density [ 1/nm^3 ]
            'probability' returns probability density
        normalize : The default is True. 
            For density=="number" it normalizes with the mean of far values (by default with the mean between distances (0.75dmax,dmax))
            for density=="probability" it normalizes with gofr.sum()
            for density=="coordination" it does nothing
            for density=="" (default) it normalizes by gmax
        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        pair_distribution : dictionary 
            having the distance and the distribution
    

        '''
        t0 = perf_counter()   
        
        #if density not in ['number','probability','max']:
            #raise NotImplementedError('density = {:s} is not Impmemented'.format(density))
            
        bins = np.arange(0,dmax+binl,binl)
        gofr = np.zeros(bins.shape[0]-1,dtype=float)
        
        if type1 is None and type2 is None:
            func = 'gofr'
            args = ()
            nc1 = self.natoms
            nc2 = (nc1-1)/2
        elif type1 is not None and type2 is None:
            func = 'gofr_type_to_all'
            fty1 = type1 == self.at_types
            nc1 = np.count_nonzero(fty1)
            nc2 = self.natoms
            args = (fty1,)
        elif type1 is None and type2 is not None:
            func ='gofr_type_to_all'
            fty2 = type2 == self.at_types
            nc1 = self.natoms
            nc2 = np.count_nonzero(fty2)
            args =(fty2,)
        elif type1==type2:
            func= 'gofr_type'
            fty = type1 == self.at_types
            nc1 =  np.count_nonzero(fty)
            nc2 =(nc1-1)/2
            args = (fty,)
        else:
            func = 'gofr_type_to_type'
            fty1,fty2 = type1 == self.at_types, type2 == self.at_types
            nc1 = np.count_nonzero(fty1)
            nc2 = np.count_nonzero(fty2)
            args= (type1 == self.at_types, type2 == self.at_types)
        
        args = (*args,bins,gofr)
        
        nframes = self.loop_trajectory(func, args)
        

        gofr/=nframes
        cb = center_of_bins(bins)
        if 'number' in density:
            
            const = 4*np.pi/3
            for i in range(0,bins.shape[0]-1):
                vshell = const*(bins[i+1]**3-bins[i]**3)
                gofr[i]/=vshell
            if 'pair' in density:
                pass
            else:
                gofr/=nc1
            if normalize:
                if 'bulk_region' in kwargs:
                    bulk_region = kwargs['bulk_region']
                else:
                    bulk_region =(0.75*dmax,dmax)
                f = np.logical_and(bulk_region[0]<cb,cb<=bulk_region[1])
                bulkv = gofr[f].mean()
                if 100*bulkv<=gofr.max():
                    raise Exception('The bulk region is very small compared to the maximum value\nTry increasing dmax')
                gofr/=bulkv
        elif density =='probability':
            gofr/=nc1*nc2
            if normalize:
                gofr/=gofr.sum()
        elif density=='coordination':
            gofr/=nc1
        elif density=='':
            if normalize:
                gofr/=gofr.max()
            
        pair_distribution = {'d':cb, 'gr':gofr }
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return pair_distribution
    
    
    def get_coords(self,frame):
        return self.timeframes[frame]['coords']
    
    def get_box(self,frame):
        return self.timeframes[frame]['boxsize']
    
    def get_time(self,frame):
        return self.timeframes[frame]['time']
    
   
    def get_whole_coords(self,frame):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.unwrap_coords(coords, box)
        return coords
    
    def bond_distance_matrix(self,ids):
        '''
        takes an array of atom ids and finds how many bonds 
        are between each atom id with the rest on the array

        Parameters
        ----------
        ids : numpy array of int

        Returns
        -------
        distmatrix : numpy array shape = (ids.shape[0],ids.shape[0])
        
        '''
        t0 = perf_counter()
        size = ids.shape[0]
        distmatrix = np.zeros((size,size),dtype=int)
        for j1,i1 in enumerate(ids):
            nbonds = self.bond_distance_id_to_ids(i1,ids)
            distmatrix[j1,:] = nbonds
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return distmatrix
    
    def bond_distance_id_to_ids(self,i,ids):
        '''
        takes an atom id and find the number of bonds between it 
        and the rest of ids. 
        If it is not connected the returns a very
        large number
        
        Parameters
        ----------
        i : int
        ids : array of int

        Returns
        -------
        nbonds : int array of same shape as ids
            number of bonds of atom i and atoms ids

        '''
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
        '''
        This function finds the minimum number of bonds
        that each atom id in ids has from args.
        If an id is not connected in any way with any of the args then
        a very large number is return within the nbonds array

        Parameters
        ----------
        ids : int array
        args : int array

        Returns
        -------
        nbonds : int array of shape as ids

        '''
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
    
    def append_timeframes(self,object2):
        '''
        Used to append one trajectory to another
        !It assumes that time of the second starts at the end of the first
        Parameters
        ----------
        object2 : A second object of the class Analysis

        Returns
        -------
        None.

        '''
        tlast = self.get_time(self.nframes-1)
        nfr = self.nframes
        for i,frame in enumerate(object2.timeframes):
            self.timeframes[nfr+i] = object2.timeframes[frame]
            self.timeframes[nfr+i]['time']+=tlast
        return

class Analysis_Confined(Analysis):
    '''
    Class for performing the analysis of confined systems
    Includes functions for Structure and Dynamics
    '''
    
    def __init__(self, trajectory_file,   
                 connectivity_file,       # itp or mol2
                 conftype,
                 topol_file = None,
                 memory_demanding=False,
                 particle_method = 'molname',
                 polymer_method = 'molname',
                 **kwargs):
        super().__init__(trajectory_file,
                         connectivity_file,
                         topol_file,
                         memory_demanding)
        self.conftype = conftype
        self.particle_method = particle_method
        self.polymer_method = polymer_method
        self.kwargs = kwargs
        
        if self.conftype =='zcylindrical':
            self.global_cylindrical_radius = kwargs['radius']
            
            
        self.confined_system_initialization()
        
        self.polymer_ids = np.where(self.polymer_filt)[0]
        self.particle_ids = np.where(self.particle_filt)[0]
        return

        
    ############## General Supportive functions Section #####################

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
    
    def find_systemic_filter(self,name):
        method = getattr(self,name+'_method')
        
        if method == 'molname':
            compare_array = self.mol_names
        elif method == 'atomtypes':
            compare_array = self.at_types
        elif method == 'molids':
            compare_array = self.mol_ids
        elif method =='atomids':
            compare_array = self.at_ids
        else:
            raise NotImplementedError('method "{}" is not Implemented'.format(method))
        
        try:
            look_name_s = self.kwargs[name]
        except KeyError:
            raise Exception('You need to provide the key word "{:s}"'.format(name))
        else:
            t1 = type(look_name_s) is str
            t2 = type(look_name_s) is list
            t3 = type(look_name_s) is np.ndarray
            if t1 or t2 or t3:
                if t2 or t3:
                    for i in range(1,len(look_name_s)):
                        if type(look_name_s[i-1]) != type(look_name_s[i]):
                                raise Exception('elements in variable {:s} must be of the same type'.format(name))
            else:
                raise NotImplementedError('{:s} variable is allowed to be either string, list or array'.format(name))
                
        filt = np.isin(compare_array,look_name_s)
        setattr(self,name+'_filt',filt)
        
        return
    def find_particle_filt(self):
        self.find_systemic_filter('particle')
        return 
    
    def find_polymer_filt(self):
        self.find_systemic_filter('polymer')
        return
       
    def translate_particle_in_box_middle(self,coords,box):
        particle_cm = self.get_particle_cm(coords)
        cds = coords.copy()
        cds += box/2 - particle_cm
        cds = implement_pbc(cds,box)
        return cds
    
    def translated_coords(self,frame):
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
        '''
        IMPORTANT Function
        This function gets the prober functions depending
        on the confinment type from a number of classes. 
        This is done in initialization to increase speed and modularity
        at the same time        

        '''
        t0 = perf_counter()
        self.find_particle_filt()
        self.nparticle = self.mol_ids[self.particle_filt].shape[0]
    
        self.find_polymer_filt()
        self.npol = self.mol_ids[self.polymer_filt].shape[0]
    
        self.particle_mass = self.atom_mass[self.particle_filt]
        self.polymer_mass = self.atom_mass[self.polymer_filt]
        
        #self.find_masses()
        self.unique_atom_types = np.unique(self.at_types)
        
        #Getting the prober functions
        self.dfun = self.get_class_function(Distance_Functions,self.conftype)
        self.box_add = self.get_class_function(Box_Additions,self.conftype)
        self.volfun = self.get_class_function(bin_Volume_Functions,self.conftype)
        self.centerfun = self.get_class_function(Center_Functions,self.conftype)
        self.unit_vectorFun = self.get_class_function(unit_vector_Functions,self.conftype)
        ##########
        
        self.find_args_per_residue(self.polymer_filt,'chain_args')
        self.find_connectivity_per_chain()
        self.find_args_per_residue(self.particle_filt,'particle_args')
        self.nparticles = len(self.particle_args.keys())
        
        self.all_args = np.arange(0,self.natoms,1,dtype=int)
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return
       
    def get_class_function(self,_class,fun,inplace=False):
        fun = getattr(_class,fun)
        if inplace:
            attr_name = _class+'_function'
            setattr(self,attr_name, fun)
        return fun
     
    def get_frame_basics(self,frame):
        #coords = self.translated_coords(frame)
        coords = self.get_coords(frame)
        box  = self.get_box(frame)          
        cm = self.get_particle_cm(coords)
        d = self.get_distance_from_particle(coords[self.polymer_filt])
        return coords,box,d,cm
    
    def get_whole_frame_basics(self,frame):
        coords = self.get_whole_coords(frame)
        box  = self.get_box(frame)          
        cm = self.get_particle_cm(coords)
        d = self.dfun(self,coords,cm)
        return coords,box,d,cm



    def get_particle_cm(self,coords):
        '''
        gets particles center of mass

        Parameters
        ----------
        coords : system coordinates

        Returns
        -------
        cm : center of mass (the number of coordinates depends on the confinment type)

        '''
        cm =self.centerfun(CM( coords[self.particle_filt], 
                               self.particle_mass))
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
            ids = self.connectivity_per_type
        elif inter == 3: 
            ids = self.angles_per_type
        elif inter == 4: 
            ids = self.dihedrals_per_type
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
            ids = self.dihedrals_per_type
        if keyword in ['3',3,'1-3']:
            ids = self.angles_per_type
        if keyword in ['2',2,'1-2']:
            ids = self.connectivity_per_type
        ids1 = np.empty(0,dtype=int)
        ids2 = np.empty(0,dtype=int)
        for k,i in ids.items():
            if k in exclude:
                continue
            ids1 = np.concatenate( (ids1,i[:,0]) )
            ids2 = np.concatenate( (ids2,i[:,-1]) )
        return ids1,ids2
    
    def find_vector_ids(self,topol_vector,exclude=[]):
        '''
        

        Parameters
        ----------
        topol_vector : list of atom types, or int for e.g. 1-2,1-3,1-4 vectors
            Used to find e.g. the segmental vector ids
        exclude : TYPE, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        ids1 : int array of atom ids
        ids2 : int array of atom ids

        '''
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
   


   
    def is_bridge(self,coords,istart,iend,periodic_image_args):
        # used to distinguish loops from bridges
        
        # check if one belong to the periodic_images and other not --> this means bridge
        e = iend in periodic_image_args 
        s = istart in periodic_image_args
        if (e and not s) or (s and not e):
            return True
        
        # check if belong to different particle
        if self.nparticles !=1:
            logger.warning('WARNING Function {:s}: This Function was never examined in test cases of more than one nanoparticles'.format(inspect.currentframe().f_code.co_name))
            r0 = coords[istart]
            re = coords[iend]
            CMps = np.empty((self.nparticles,3),dtype=float)
            for i,(k,args) in enumerate(self.particle_args.items()):
                CMps[i] = CM(coords[args],self.atom_mass[args])
            
            if not e and not s: # if in main box no need to check periodic images
                rer = CMps-re
                r0r = CMps -r0
                de = np.sum(rer*rer,axis=1)
                d0 = np.sum(r0r*r0r,axis=1)
            else:
                de = 1e16 ; d0=1e16
                for L in self.box_add(self.get_box(self.current_frame)):
                    r0r = CMps+L -r0
                    rer = CMps+L -re
                    de = np.minimum(de,np.sum(rer*rer,axis=1))
                    d0 = np.minimum(d0,np.sum(r0r*r0r,axis=1))
            
            if de.argmin()!=d0.argmin():
                return True
                
        return False
    
    def get_filt_train(self,dads,coords,box):
        '''
        

        Parameters
        ----------
        dads : float. below this distance everything is adsorbed
        coords : system coords
        box : systems box

        Returns
        -------
        ftrain : bool array (shape = coords.shape[0])
        image_trains : bool_array() (shape = coords.shape[0])

        '''
        ftrain = False
       
        for L in self.box_add(box):
            cm = self.get_particle_cm(coords+L)
            d = self.dfun(self,coords,cm)
            ftrain = np.logical_or(ftrain,np.less_equal(d,dads))
        ftrain = np.logical_and(ftrain,self.polymer_filt)
        
        self_trains = np.less_equal(
            self.dfun(self,coords,self.get_particle_cm(coords)),dads)
        image_trains = np.logical_and(ftrain,np.logical_not(self_trains))
        
        return ftrain,image_trains
    
    def get_distance_from_particle(self,r=None,ids=None):
        '''
        Give either r or ids
        If you give r then the 
        minimum image distance of r from particle will be found
        
        Parameters
        ----------
        r : coordinates (float array shape=(n,3) ).
        ids : int array

        Returns
        -------
        d : minimum image distance (float array shape = (n,) )
            Depends on the confinment type
            e.g. for flat surfaces parallel on the xy plane 
            only the z direction is considered

        '''
        frame = self.current_frame
        
        box = self.get_box(frame)
        coords = self.get_coords(frame)
        if r is None:
            r = coords[ids]
        cm = self.get_particle_cm(coords)
        
        d = 1e16
        for L in self.box_add(box):
            d = np.minimum(d,self.dfun(self,r,cm+L))
        return d
    
    def conformations(self,dads,coords):
        '''
        Returns the ids of trains,loops,tails 
        and bridges given the coords

        Parameters
        ----------
        dads : float 
            adsorption distance
        coords : coordinates (float array shape=(n,3) )

        Returns
        -------
        ads_chains : int array
            ids of adsorbeds chains.
        args_train : int array
            ids of trains
        args_tail : int array
            ids of tails
        args_loop : int array
            ids of loops
        args_bridge : int arry
            ids of bridges

        '''
        box = self.get_box(self.current_frame)
        
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
                '''
                assert not ftrain[chunk].any(), \
                    'chunk in train chain {:d}, node {}, chunk size = {:d}\
                    ,istart = {:d}, iend ={:d} \n\n chunk =\n {} \n\n'.format(j,
                    node,chunk.shape[0],istart,iend,chunk)
                assert fads_chains[chunk].all(),\
                    'chunk out of adsorbed chains, j = {:d} ,\
                        node = {} n\n chunk \n {} \n\n'.format(j,
                        node,chunk)
                '''
                if loopBridge:
                    if not self.is_bridge(coords,istart,iend,periodic_image_args):    
                        args_loop = np.concatenate( (args_loop, chunk) )             
                    else:
                        #logger.debug('chain = {:d}, chunk | (istart,iend) = ({:d}-{:d}) is bridge'.format(j,istart,iend))
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
    
    def connected_chunks(self,args):
        '''
        takes a set of ids and finds the connected chunks from it
        It is very usefull to find conformation size distributions
        Parameters
        ----------
        args : int array of atom ids
        
        Returns
        -------
        chunks : list of sets. Each set has the ids of the connected chunks

        '''
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
        
            #aold # (for debugging-combined with assertion above)
        #ass.print_time(perf_counter()-t0,'connected_chunks')
        return chunks
    
    def length_of_connected_chunks(self,args,coords,exclude_bonds=None):
        # this currently is suitable for linear chains in corse-grained or united atom represtation
        # if there is an all atom representation exclude_bonds must be used
        bonds = self.sorted_connectivity_keys
        f1 = np.isin(bonds[:,0],args)
        f2 = np.isin(bonds[:,1],args)
        f = np.logical_and(f1,f2)
        cb = bonds[f]
        coords1 = coords[cb[:,0]]
        coords2 = coords[cb[:,1]]
        r = coords2-coords1
        dists = np.sqrt(np.sum(r*r,axis=1))
        return dists.sum()
    ############### End of Conformation Calculation Supportive Functions #####
    
    ######## Main calculation Functions for structural properties ############      
    
    def calc_density_profile(self,binl,dmax,
                             mode='mass',option='',flux=None,**kwargs):
        '''
        Master function to compute the density profile

        Parameters
        ----------
        binl : float
            binningl legth
        dmax : float
            maximum distance to calculate
        mode : string
            'mass' for mass density
            'number' for number density
            For conformations option is set automatically to massnumber
        option : string
            'pertype' decomposes the density to each atomic type contribution
            '2side' computes the profile to both sides.
                Only valid for one directional confinement (i.e. zdir,ydir,xdir) 
        flux : bool
            if True it calculates density fluxuations
            by first calling this function to compute the density.
            density fluctuations are computed by the standard deviation

        Returns
        -------
        density_profile : dictionary containing 
            discription :  key  |  value
            the distance: 'd'  | float array 
            the density: 'rho' | float array
            density of  'type1' | float array1
            density of 'type2' | float array2
                          .
                          .
                          .
            density of 'typen' | float arrayn
            density fluctuations: 'rho_flux' | float array

        '''
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
        mass_pol = self.atom_mass[self.polymer_filt] 
        
        
        if option =='':
            args = (nbins,bins,rho)
            contr=''
        elif option == 'pertype':
            rho_per_atom_type = {t:np.zeros(nbins,dtype=float) for t in self.unique_atom_types }
            ftt = {t: t == self.at_types[self.polymer_filt] for t in rho_per_atom_type.keys() }
            args = (nbins,bins,rho,rho_per_atom_type,ftt)
            contr ='__pertype'
        elif option =='2side':
            rho_down = np.zeros(nbins,dtype=float)
            args =(nbins,bins,rho,rho_down)
            contr ='__2side'
        elif option =='conformations':              
            confdens = {nm+k:np.zeros(nbins,dtype=float) for nm in ['m','n']
                        for k in ['rho','train','tail','loop','bridge','free'] 
                        }         
            stats = { k : 0 for k in ['adschains','train','looptailbridge',
                                  'tail','loop','bridge']}
            dlayers = [(bins[i],bins[i+1]) for i in range(nbins)]
            args = (kwargs['dads'],dlayers, confdens, stats)
            contr='__conformations'
            mode='massnumber'
            
        if mode =='mass': args =(*args,mass_pol)
        
        
        if flux is not None and flux !=False:
            density_profile.update(self.calc_density_profile(binl,dmax,mode,option=option,**kwargs))
            rho_mean = density_profile['rho'].copy()
            rho_mean/=scale
            args = (nbins,bins,rho,mass_pol,rho_mean**2)
            func =  'mass'+'_density_profile'+'_flux'
            if mode !='mass' or option!='':
                logger.warning('mass mode and total density fluxuations are calculated')
        else:
            func =  mode+'_density_profile'+contr    
        
        
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
            if option =='pertype':
                for t,rhot in rho_per_atom_type.items(): 
                    density_profile[t] = rhot*scale/nframes 
            elif option =='2side':
                rho_down *=scale/nframes
                density_profile.update({'rho_up':rho,'rho_down':rho_down,'rho':0.5*(rho+rho_down)})
            elif option =='conformations':
                stats = {k+'_perc':v/self.npol/nframes if 'adschains'!=k else v/len(self.chain_args)/nframes
                         for k,v in stats.items()}
                dens= {k:v*scale/nframes for k,v in confdens.items()}

                for k in ['train','tail','loop','bridge','free']: 
                    dens['mrho'] += dens['m'+k] 
                    dens['nrho'] += dens['n'+k]
                    
                dens['rho'] = dens['mrho']
                density_profile.update(dens)
                density_profile.update({'stats':stats})
                
        #############
        
        tf = perf_counter() -t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return density_profile
    

    
    def calc_P2(self,binl,dmax,topol_vector,option='',**kwargs):
        '''
        Calculates P2 bond parameter
        The unit vector is predifined by the type of confinement
        
        Parameters
        ----------
        binl : float
            binning legth
        dmax : float
            maximum distance.
        topol_vector : int or list of atom types
            int is 2,3,4 for 1-2, 1-3, 1-4 bond vectors respectively
            with a list of atom types it will look 
            the connectivity for len(topol_vector) ==2
            the angles for len(topol_vector) ==3
            the dihedrals for len(topol_vector) ==4
            and it will get the given vectors

        Returns
        -------
        orientation : dictionary containing
        discription :  key  |  value
        the distance: 'd'  | float array
        the function: 'P2' | float array
        '''
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
        costh_unv = [[] for i in range(len(dlayers))]
        costh = np.empty(nvectors,dtype=float)
        if option in ['bridge','loop','train','tail','free']:
            args = (ids1,ids2,dlayers,kwargs['dads'],option,costh,costh_unv)
            s = '_conformation'
        elif option=='':
            s =''
            args = (ids1,ids2,dlayers,costh,costh_unv)
        else:
            raise NotImplementedError('option "{}" not Implemented.\n Check your spelling when you give strings'.format(option))
        nframes = self.loop_trajectory('P2'+s, args)

       
        costh2_mean = np.array([ np.array(c).mean() for c in costh_unv ])
        costh2_std  = np.array([ np.array(c).std()  for c in costh_unv ])
        s='P2'
        orientation = {'d':  d_center} 
        orientation.update({s: 1.5*costh2_mean-0.5, s+'(std)' : 1.5*costh2_std-0.5 })
        
        tf = perf_counter() - t0
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return orientation   
        
    def calc_particle_size(self):
        t0 = perf_counter()
        part_size = np.zeros(3)
        args = (part_size,)
        nframes = self.loop_trajectory('particle_size',args)
        part_size /= nframes
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return part_size
        
    def calc_dihedral_distribution(self,phi,dads=None,filters=dict()):
        '''
        Computes dihedral distributions as a function of given populations
        ----------
        dads : float
          adsorption distance
        dmax : float
            maximum distance
        binl : float
            binning length

        Returns
        -------
        dihedral_distr : dictionary

        '''
        t0 = perf_counter()
        diht,ft = self.calc_dihedrals_t(phi,dads=dads,filters = filters)
        distrib = {k: [] for filt in filters.values() for k in filt }
        
        for k in distrib:
            for t,dih in diht.items():
                distrib[k].extend(dih[ft[k][t]])
        
        distrib['system'] = []
        for t,dih in diht.items():
            distrib['system'].extend(dih)
            
        for k in distrib:
            distrib[k] = np.array(distrib[k])*180/np.pi
        
        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return distrib
    
    def calc_chain_characteristics(self,dmin,dmax,binl):
        '''
        Chain characteristics as a function of distance of chain center of mass
        Parameters
        ----------
        dmin : float
            minimum distance
        dmax : float
            maximumum distance
        binl : float
            binning length

        Returns
        -------
        chain_chars : dictionary with keys the characteristics in strings
            and values float arrays. Also the distance is included

        '''
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
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return chain_chars
    
    ###### End of Main calculation Functions for structural properties ########
    
    def vects_per_t(self,ids1,ids2,
                         filters={},
                         dads=0):
        '''
        Takes two int arrays  of atom ids and finds 
        the vectors per time between them

        Parameters
        ----------
        ids1 : int array of atom ids 1
        ids2 : int array of atom ids 2
        filters : dictionary of filters. See filter documentation
        dads : float
            adsorption distance

        Returns
        -------
        vec_t : dictionary of keys the time and values the vectors in numpy array of shape (n,3)
        filt_per_t : dictionary of keys the filt name and values
            dictionary of keys the times and boolian arrays

        '''
        t0 = perf_counter()
        vec_t = dict()
        filt_per_t = dict()
        
        args = (ids1,ids2,filters,dads,vec_t,filt_per_t)
        
        nframes = self.loop_trajectory('vects_t', args)
                
        filt_per_t = ass.rearrange_dict_keys(filt_per_t)
        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return vec_t,filt_per_t

    def chains_CM(self,coords):
        '''
        Given the coordinates finds the chains center of mass

        Parameters
        ----------
        coords : system coordinates

        Returns
        -------
        chain_cm : float array (nchains,3)

        '''
        chain_arg_keys = self.chain_args.keys()
        
        chain_cm = np.empty((len(chain_arg_keys),3),dtype=float)
        
        for i,args in enumerate(self.chain_args.values()):
            chain_cm[i] = CM(coords[args],self.atom_mass[args])
        
        return chain_cm

    def segs_CM(self,coords,segids):
        '''
        
        Given the coordinates and the segmental ids finds the segmental center of mass
        
        Parameters
        ----------
        coords : system coordinates 
        segids : int array (nsegments,nids_per_segment)

        Returns
        -------
        segcm : float array (nsegments,3)

        '''
        n = segids.shape[0]
        segcm = np.empty((n,3),dtype=float)
        numba_CM(coords,segids,self.atom_mass,segcm)
        return segcm
            

    def calc_dihedrals_t(self,dih_type,
                             dads=0,filters={'all':None}):
        '''
        Dihedrals per time

        Parameters
        ----------
        dih_type : list of 4 strings
            dihedral type
        dads : float
            adsorption distance
        filters : dictionary
            see filters documentation.

        Returns
        -------
        dihedrals_t : dictionary 
        keys: times
        values: float array (ndihedrals,)
        filt_per_t : see filters documentation

        '''
        t0 = perf_counter()
        
        dihedrals_t = dict()
        filt_per_t = dict()
       
        dih_ids = self.dihedrals_per_type[dih_type] #array (ndihs,4)
        ids1 = dih_ids[:,0] ; ids2 = dih_ids[:,3] 

        args = (dih_ids,ids1,ids2,filters,dads,dihedrals_t,filt_per_t)
        
        nframes = self.loop_trajectory('dihedrals_t', args)
        
        filt_per_t = ass.rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return dihedrals_t,filt_per_t
    


    def calc_Ree_t(self,filters={},
                       dads=None):
        '''
        End to End vectors as a function of time

        Parameters
        ----------
        filters : dictionary
            see filter documentation
        dads : adsorption distance
            adsorption distance

        Returns
        -------
        Ree_t : dictionary
            keys times
            values: float arrays (nchains,3)
        filt_per_t : dictionary
            see filter documentation

        '''
        
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
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
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
        '''
        Chain dipole moment as a function of t

        Parameters
        ----------
        filters : dictionary
            see filter documentation.
        dads : float
            adsorption distance.

        Returns
        -------
        dipoles_t : dictionary
            keys: times
            values: float array (nchains,3)
        filters_t : dictionary
            see filter documentation

        '''
        t0 = perf_counter()
        
        filters = {'chain_'+k : v for k,v in filters.items()}
        
        self.set_partial_charge()

        
        dipoles_t = dict()
        filters_t = dict()
        args = (filters,dads,dipoles_t,filters_t)
        
        nframes = self.loop_trajectory('chain_dipole_moment',args)
        
        filters_t = ass.rearrange_dict_keys(filters_t)
        
        tf = perf_counter() - t0
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dipoles_t,filters_t
    
    def find_segmental_ids(self,ids1,ids2,segbond):
        '''
        Used to find the segmental ids
        Works for linear polymers at the moment
        
        Parameters
        ----------
        ids1 : int array
            atom ids 1
        ids2 : int array
            atom ids 2
        segbond : tuple of two stings 
            the type of bond that connects the segments 
        Returns
        -------
        seg_ids_numpy : int array (nsegments,nids_per_segment)
            an array with the ids of each segment

        '''
        t0 = perf_counter()
        conn_cleaned = {k:v for k,v in self.connectivity.items()
                        if v != segbond
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
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return  seg_ids_numpy
    
    def calc_segmental_dipole_moment_t(self,topol_vector,
                                       segbond,filters={'all':None},
                                       dads=1):
        '''
        Calculates dipole moment vectors as a function of time


        Parameters
        ----------
        topol_vector : list of 4 strings
            finds the segment edge ids
        segbond : tuple of 2 strings
            the bond type connecting the segments
        filters : dictionary
            see filter documentation
        dads : float
            adsorption distance
        Returns
        -------
        dipoles_t : dictionary
            keys: times
            values: float array (nvector,3)
        filters_t : dictionary
            see filter documentatin

        '''
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
        
        filters_t = ass.rearrange_dict_keys(filters_t)
        
        tf = perf_counter() - t0
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dipoles_t,filters_t
    
    def calc_segmental_dipole_moment_correlation(self,topol_vector,
                                       segbond,filters={'all':None},
                                       dads=1.025):
        '''
        Computes static correlation between segments
        
         Parameters
        ----------
        topol_vector : list of 4 strings
            finds the segment edge ids
        segbond : tuple of 2 strings
            the bond type connecting the segments
        filters : dictionary
            see filter documentation
        dads : float
            adsorption distance

        Returns
        -------
        correlations : dictionary
            keys   : depent on filter 
            values : dictionary
                    keys: number of bonds connecting the segments
                    values: static correlation

        '''
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
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
    
        return correlations
    
    def calc_chainCM_t(self,filters={'all':None}, dads=1,option=''):
        '''
        Computes chains center of mass as a function of time
        Parameters
        ----------
        filters : dictionary
            see filter documentation
        dads : float
            adsorption distance
        option : string
            for feature use
        coord_type : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        vec_t : dictionary
            keys: times
            values: float array (nchains,3)
        filt_per_t : dictionary
            see filter documentation

        '''
        t0 = perf_counter()
        filters = {'chain_'+k: v for k,v in filters.items()} #Need to modify when considering chains
        
        vec_t = dict()
        filt_per_t = dict()
        
        args = (filters,dads,vec_t,filt_per_t)
        
        nframes = self.loop_trajectory('chainCM_t'+option, args)
      
        filt_per_t = ass.rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return vec_t,filt_per_t
    
    def calc_segCM_t(self,topol_vector,segbond,
                     filters={'all':None}, dads=1,option=''):
        '''
        Returns the segment center of mass as a function of time

        Parameters
        ----------
        topol_vector : list of 4 strings
            finds the segment edge ids
        segbond : tuple of 2 strings
            the bond type connecting the segments
        filters : dictionary
            see filter documentation
        dads : float
            adsorption distance
        option: string
            for feature use

        Returns
        -------
        vec_t : dictionary
            keys: times
            values: float array (nsegments,3)
        filt_per_t : dictionary
            see filter documentation

        '''
        t0 = perf_counter()
     
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        segmental_ids = self.find_segmental_ids(ids1, ids2, segbond)
        
        vec_t = dict()
        filt_per_t = dict()
        
        args = (filters,dads,ids1,ids2,segmental_ids,vec_t,filt_per_t)
        
        nframes = self.loop_trajectory('segCM_t'+option, args)
      
        filt_per_t = ass.rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return vec_t,filt_per_t
    
    def calc_conformations_t(self,dads,option=''):
        '''
        computes conformations as a function of time

        Parameters
        ----------
        dads : float
            adsorption distance.
        option : string
            for feature use

        Returns
        -------
        confs_t : dictionary
            keys: strings (conformations)
            values: dictionary
                keys: times
                values: number of atoms on the specific conformation

        '''
        t0 = perf_counter()
        
        confs_t = dict()
        args = (dads,confs_t)
        nframes = self.loop_trajectory('confs_t'+option, args)
        confs_t = ass.rearrange_dict_keys(confs_t)
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
    
        return confs_t
    
    def calc_segmental_vectors_t(self,topol_vector,filters={},
                                     dads=0):
        '''
        Calculates 1-2,1-3 or 1-4 segmental vectors as a function of time
        Parameters
        ----------
        topol_vector : list of atom types, or int for e.g. 1-2,1-3,1-4 vectors
            Used to find e.g. the segmental vector ids
        filters : dictionary.
            see filters documentation
        dads : float
            adsorption distance

        Returns
        -------
        segvec_t : dictionary
            keys: times
            values: float array (nsegments,3)
        filt_per_t : dictionary
            see filter documentation
        '''
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        t0 = perf_counter()
        
        segvec_t, filt_per_t = self.vects_per_t(ids1, ids2, 
                                                         filters,
                                                         dads)
        
        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return segvec_t,filt_per_t
    
    def calc_adsorbed_segments_t(self,topol_vector,dads):
        '''
        Calculates adsorbed segments as a function of time

        Parameters
        ----------
        topol_vector : list of atom types, or int for e.g. 1-2,1-3,1-4 vectors
            Used to find e.g. the segmental vector ids
        dads : float
            adsorption distance.

        Returns
        -------
        dictionary
            keys: times
            values: bool array (nsegments,)

        '''
        t0 = perf_counter()
        
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        segvec_t, adsorbed_segments_t = self.vects_per_t(ids1, ids2,
                                                         filters={'space':(0,dads)})
        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return adsorbed_segments_t[(0,dads)]
    
    
    def init_xt(self,xt,dtype=float):
        '''
        takes a dictionary of keys times and values some array
        and converts it to numby

        Parameters
        ----------
        xt : dictionary
            keys: times
            values: array
        dtype : type
            type of the array.

        Returns
        -------
        x_nump : numpy array of type dtype

        '''
        x0 =xt[0]
        nfr = len(xt)
        shape = (nfr,*x0.shape)
        x_nump = np.empty(shape,dtype=dtype)
        
        for i,t in enumerate(xt.keys()):
           x_nump[i] = xt[t]
        
        return  x_nump
    
    def init_prop(self,xt):
        '''
        Allocates the arrays to fill the dynamical or kinetic property
        that will be calculated

        Parameters
        ----------
        xt : dictionary
            keys: times
            values: array
            It is the input.

        Returns
        -------
        Prop_nump : float array
        nv : float array, is used as a counter
        '''
        nfr = len(xt)
        Prop_nump = np.zeros(nfr,dtype=float)
        nv = np.zeros(nfr,dtype=float)
        return Prop_nump,nv
  
    def get_Dynamics_inner_kernel_functions(self,prop,filt_option,weights_t):
        '''
        gets the options to set up the prober functions for
        the Dynamics kernel

        Parameters
        ----------
        prop : string
            P1: p1 dynamics
            P2: p2 dynamics
            MSD: mean square displacement
            
        filt_option : string
            simple: consider the population based on time origin
            strict: consider the population based on time origin and  time t
            change: consider the population based on time origin and changed on time t
        weights_t : dictionary
            keys: time
            values: float array (total population,)
        Returns
        -------
        funcs : the main kernel function, 
                the function which sets the arguments
                the inner function

        '''
        mapper = {'p1':'costh_kernel',
                  'p2':'cos2th_kernel',
                  'msd':'norm_square_kernel'}
        inner_func_name = mapper[prop.lower()] 
        
        name = 'dynprop'
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
        
        logger.info(' func name : "{:s}" \n argsFunc name : "{:s}" \n innerFunc name : "{:s}" '.format(func_name,args_func_name,inner_func_name))
        
        funcs = (globals()[func_name],
                 globals()[args_func_name],
                 globals()[inner_func_name])
        
        return funcs
  
    def Dynamics(self,prop,xt,filt_t=None,weights_t=None,
                 filt_option='simple', block_average=False):
        '''
        

        Parameters
        ----------
        prop : string
            P1: p1 dynamics
            P2: p2 dynamics
            MSD: mean square displacement
        xt : dictionary
            keys: times
            values: vector array (total population,nd) where nd =1,2,3
            It is the input.

        filt_t : dictionary
            keys: times
            values: boolean array
        weights_t : dictionary
            keys: time
            values: float array (total population,)
        filt_option : string
            simple: consider the population based on time origin
            strict: consider the population based on time origin and  time t
            change: consider the population based on time origin and changed on time t

        block_average : bool
            Ways of averaginf
            True  <  < >|population  >|time origins
            False < >|population,time origins

        Returns
        -------
        dynamical_property : dictionary
            keys: time
            values: float

        '''
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
        
        
        func,func_args,func_inner = \
        self.get_Dynamics_inner_kernel_functions(prop,filt_option,weights_t)
        
        args = (func,func_args,func_inner,
                Prop_nump,nv,
                x_nump,f_nump,w_nump,
                block_average)
        
        prop_kernel = globals()['DynamicProperty_kernel']
        overheads = perf_counter() - tinit
        
        try:
            #prop_kernel(prop_nump, nv, x_nump, f_nump, nfr)
            prop_kernel(*args)
        except ZeroDivisionError as err:
            logger.error('Dynamics Run {:s} --> There is a {} --> Check your filters or weights'.format(prop,err))
            return None
        
       
        tf2 = perf_counter()
        if prop.lower() !='p2':
            dynamical_property = {round(t,4):p for t,p in zip(xt,args[3])}
        else:
            dynamical_property = {round(t,4):0.5*(3*p-1) for t,p in zip(xt,args[3])}
        tf3 = perf_counter() - tf2
        
        tf = perf_counter()-tinit
        #logger.info('Overhead: {:s} dynamics computing time --> {:.3e} sec'.format(prop,overheads+tf3))
        ass.print_time(tf,inspect.currentframe().f_code.co_name +'" ---> Property: "{}'.format(prop))
        return dynamical_property
    
    def get_Kinetics_inner_kernel_functions(self,wt):
        '''
        

        Parameters
        ----------
        wt :  dictionary
            keys: time
            values: float array (total population,)

        Returns
        -------
        returns function to call in kinetics kernel,
                function to set the arguments
        

        '''
        if wt is None:
            func_args = 'get__args'
            func_name = 'Kinetics_inner__kernel'
        else:
            func_args = 'get_weighted__args'
            func_name = 'Kinetics_inner_weighted__kernel'
        
        logger.info('func name : {:s} , argsFunc name : {:s}'.format(func_name,func_args))
        
        return globals()[func_name],globals()[func_args]
    
    def Kinetics(self,xt,wt=None,block_average=False):
        '''

        Parameters
        ----------
        xt : dictionary
            keys:times
            values: boolean array (total population,)
        wt : dictionary
            keys: time
            values: float array (total population,)
        block_average : bool
            Ways of averaginf
            True  <  < >|population  >|time origins
            False < >|population,time origins

        Returns
        -------
        kinetic_property :dictionary
            keys: time
            values: float (it represents the percentage of state change)
        '''
        tinit = perf_counter()
        
        Prop_nump,nv = self.init_prop(xt)
        x_nump = self.init_xt(xt,dtype=bool)
        
        if wt is not None:
            w_nump = self.init_xt(wt)
        else:
            w_nump = None
        
        func_name,func_args = self.get_Kinetics_inner_kernel_functions(wt)
        
        args = (func_name,func_args,
                Prop_nump,nv,
                x_nump,w_nump,
                block_average)
        Kinetics_kernel(*args)
        
        kinetic_property = {t:p for t,p in zip(xt,Prop_nump)}
        tf = perf_counter()-tinit
        #logger.info('Overhead: {:s} dynamics computing time --> {:.3e} sec'.format(prop,overheads+tf3))
        ass.print_time(tf,inspect.currentframe().f_code.co_name +'" ---> Property: "Kinetics')
        return kinetic_property
    
    def get_TACF_inner_kernel_functions(self,prop,filt_option,weights_t):
        '''
        

        Parameters
        ----------
        prop : string
            cos: cos function
            sin: sin function

        filt_t : dictionary
            keys: times
            values: boolean array
        weights_t: dictionary
            keys: time
            values: float array (total population,)
        Returns
        -------
        funcs : 7 functions for the TACF kernel

        '''
        inner_mapper = {'cos':'cosCorrelation_kernel',
                        'sin':'sinCorrelation_kernel'}
        inner_zero_mapper = {'cos':'fcos_kernel',
                             'sin':'fsin_kernel'}
        
        inner_func_name = inner_mapper[prop.lower()] 
        inner_zero_func_name = inner_zero_mapper[prop.lower()]
        
        name = 'dynprop'
        af_name = 'get'
        args_z_name = 'get_zero'
        mean_func_name = 'mean'
        secmom_func_name = 'secmoment'
        
        if filt_option is not None:
            ps = '_{:s}'.format(filt_option)
            pz = '_filt'
            name += ps 
            af_name+= ps
            args_z_name +=pz 
            mean_func_name+=pz
            secmom_func_name+=pz
        if weights_t is not None:
            ps = '_weighted'
            args_z_name += ps
            name += ps
            af_name += ps
            mean_func_name+=ps
            secmom_func_name+=ps
        
        func_name = '{:s}__kernel'.format(name)
        args_func_name = '{:s}__args'.format(af_name)
        args_zero_func_name = '{:s}__args'.format(args_z_name)
        mean_func_name = '{:s}__kernel'.format(mean_func_name)
        secmom_func_name = '{:s}__kernel'.format(secmom_func_name)
        
        func_names = [func_name,args_func_name, inner_func_name,
                      mean_func_name,secmom_func_name,
                      args_zero_func_name,
                      inner_zero_func_name]
        
        s = ''.join( ['f{:d}={:s} \n'.format(i,f) for i,f in enumerate(func_names)] )
        
        logger.info(s)
        
        funcs = tuple([ globals()[f] for f in func_names])
        return funcs
  
    def TACF(self,prop,xt,filt_t=None,
             wt=None,filt_option=None,block_average=False):
        '''
        

        Parameters
        ----------
        prop : string
            cos: cos function
            sin: sin function
        xt : TYPE
            DESCRIPTION.
        filt_t : dictionary
            keys: times
            values: boolean array
        wt: dictionary
            keys: time
            values: float array (total population,)
        filt_option : string
            simple: consider the population based on time origin
            strict: consider the population based on time origin and  time t
            change: consider the population based on time origin and changed on time t
        block_average : bool
            Ways of averaginf
            True  <  < >|population  >|time origins
            False < >|population,time origins

        Returns
        -------
        TACF_property : dictionary
            keys: time
            value: float 

        '''
        tinit = perf_counter()
        
        Prop_nump,nv = self.init_prop(xt)
        mu_val,mu_num = self.init_prop(xt)
        secmom_val,secmom_num = self.init_prop(xt)
        
        x_nump = self.init_xt(xt,dtype=float)
        
        if filt_t is not None:
            f_nump = self.init_xt(filt_t,dtype=bool)
            if filt_option is None:
                filt_option= 'simple'
        else:
            f_nump = None
            filt_option = None
        
        if wt is not None:
            w_nump = self.init_xt(wt)
        else:
            w_nump = None
        
        func_name, func_args, inner_func,\
        mean_func, secmoment_func, func_args_zero, inner_func_zero \
        = self.get_TACF_inner_kernel_functions(prop,filt_option,wt)
        

        args = (func_name, func_args, inner_func,
              mean_func, secmoment_func, func_args_zero, inner_func_zero,
              Prop_nump,nv,
              mu_val,mu_num,secmom_val,secmom_num,
              x_nump, f_nump, w_nump,
              block_average)
    
        TACF_kernel(*args)
        #print(Prop_nump)
        TACF_property = {t:p for t,p in zip(xt,Prop_nump)}
        tf = perf_counter()-tinit
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        
        return TACF_property
    

class Filters():
    '''
    The Filters class contains the functions that produce
    boolean arrays ("filters") that are used to distinguish the 
    population, e.g. segments to trains, loops ..., chains at distance etc ...
    '''
    def __init__(self):
        pass
    
    @staticmethod
    def calc_filters(self,filters,*args):
        bool_data = dict()
        
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
    def space(self,layers,ids1,ids2,coords,*args):
        #coords = self.get_whole_coords(self.current_frame)

        d1 = self.get_distance_from_particle(ids = ids1)
        d2 = self.get_distance_from_particle(ids =ids2)
        
        f1 = Filters.filtLayers(layers,0.5*(d1+d2))
        return f1
    
    @staticmethod
    def filtLayers(layers,d):
        if ass.iterable(layers):
            if ass.iterable(layers[0]):
                return {dl : filt_uplow(d , dl[0], dl[1]) for dl in layers}
            else:
                return {layers: filt_uplow(d , layers[0], layers[1])}
        return dict()
    
    @staticmethod
    def filtLayers_inclucive(layers,d):
        if ass.iterable(layers):
            if ass.iterable(layers[0]):
                return {dl : filt_uplow_inclucive(d , dl[0], dl[1]) for dl in layers}
            else:
                return {layers: filt_uplow_inclucive(d , layers[0], layers[1])}
        return dict()
    
    @staticmethod
    def BondsTrainFrom(self,bondlayers,ids1,ids2,coords,dads,*args):
        #t0 = perf_counter()
        
        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations(dads,coords)
        
        args_rest_train = np.concatenate( (args_tail,args_loop,args_bridge ) )
        nbonds1 = self.ids_nbondsFrom_args(ids1,args_rest_train)
        nbonds2 = self.ids_nbondsFrom_args(ids2,args_rest_train)
        nbonds = np.minimum(nbonds1,nbonds2)
        
        return Filters.filtLayers_inclucive(bondlayers,nbonds)
    
    def BondsFromEndGroups(self,bondlayers,ids1,ids2,coords,dads,*args):
        #t0 = perf_counter()
        args_endGroups = self.get_EndGroup_args()
        
        nbonds1 = self.ids_nbondsFrom_args(ids1,args_endGroups)
        nbonds2 = self.ids_nbondsFrom_args(ids2,args_endGroups)
        nbonds = np.minimum(nbonds1,nbonds2)
        
        return Filters.filtLayers_inclucive(bondlayers,nbonds)
    
    @staticmethod
    def BondsFromTrain(self,bondlayers,ids1,ids2,coords,dads,*args):
        #t0 = perf_counter()
        
        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations(dads,coords)
        
        nbonds1 = self.ids_nbondsFrom_args(ids1,args_train)
        nbonds2 = self.ids_nbondsFrom_args(ids2,args_train)
        nbonds = np.minimum(nbonds1,nbonds2)
        
        return Filters.filtLayers_inclucive(bondlayers,nbonds)

    @staticmethod
    def conformationDistribution(self,fconfs,ids1,ids2,coords,dads,*args):
        
        
        ads_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations(dads,coords)
       
        filt = dict()
        for conf,intervals in fconfs.items():
            
            args = locals()['args_'+conf]
            
            connected_chunks = self.connected_chunks(args)
            
            sizes = np.array([chunk.__len__() for chunk in connected_chunks])
            
            
            filt['{}:distr'.format(conf)] = sizes
            
            for inter in intervals:
                
                chunk_int =set()
                for chunk, size in zip(connected_chunks,sizes):
                    if inter[0]<=size<inter[1]:
                        chunk_int = chunk_int | chunk
                        
                args_chunk = np.array(list(chunk_int),dtype=int)
                f = Filters.filt_bothEndsIn(ids1, ids2, args_chunk)
                
                filt['{}:{}'.format(conf,inter)] = f
                
        return filt
    
    @staticmethod
    def conformations(self,fconfs,ids1,ids2,coords,dads,*args):
        
        ads_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations(dads,coords)
        
        all_not_free = np.concatenate((args_train,args_tail,args_loop,args_bridge))
        
        all_args = self.all_args
        args_free = all_args [ np.logical_not( np.isin(all_args, all_not_free) ) ]

        filt = dict()
        if ass.iterable(fconfs):
            for conf in fconfs:
                conf_args = locals()['args_'+conf]
                filt[conf] = Filters.filt_bothEndsIn(ids1, ids2, conf_args)
        elif type(fconfs) is str:
            conf_args = locals()['args_'+fconfs]
            filt[fconfs] = Filters.filt_bothEndsIn(ids1, ids2, conf_args)
        #ass.print_time(perf_counter()-t0,inspect.currentframe().f_code.co_name)
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
    def chain_space(self,layers,chain_cm,*args):
        d = self.get_distance_from_particle(chain_cm)
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
    def get_ads_degree(self,coords,dads):
        chain_arg_keys = self.chain_args.keys()
        cads = np.empty(len(chain_arg_keys),dtype=bool)
        degree = np.empty(cads.shape[0],dtype=float)
        d = self.get_distance_from_particle(coords)
        fd = filt_uplow(d,0,dads)
        for i,args in enumerate(self.chain_args.values()):   
            f = fd[args]
            cads[i] = f.any()
            degree[i] = np.count_nonzero(f)/f.shape[0]
        
        return degree,cads
    @staticmethod
    def chain_adsorption(self,ads_degree, chain_cm, coords, dads,*args):
        
        
        degree,cads = Filters.get_ads_degree(self,coords,dads)
            
        filt_ads = dict()
        
        filt_ads.update( Filters.filtLayers(ads_degree,degree) )
        filt_ads.update({'ads':cads,'free':np.logical_not(cads),'degree':degree})
        
        return filt_ads
    
class coreFunctions():
    '''
    This class contains all functions that are called during 
    looping the trajectory frames. In this way the main functions are
    more compact and we do not have to rewrite the same looping
    method for each property. This functions fill dictionaries or arrays 
    allocated from the function that they
    were called (Usuallay starting with the word 'calc_')
    '''
    def __init__():
        pass
    @staticmethod
    def atomic_coordination(self,maxdist,args1,args2,coordination):
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords1 = coords[args1]
        coords2 = coords[args2]
        numba_coordination(coords1,coords2,box,maxdist,coordination)
        return
                
    @staticmethod
    def particle_size(self,part_size):
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.translate_particle_in_box_middle(coords,box)
        part_coords = coords[self.particle_filt]
        part_s = part_coords.max(axis = 0 ) - part_coords.min(axis = 0 )
        #logger.debug('frame = {:d} --> part size = {} '.format(frame,part_s))
        part_size += part_s
        return 
    
    @staticmethod
    def theFilt(self,filters,dads,ids1,ids2,filt_per_t):
        frame = self.current_frame
        coords = self.get_whole_coords(frame)
        
        time = self.get_time(frame)
       
        if frame == self.first_frame:
            self.time_zero=time
        
        key = self.get_timekey(time,self.time_zero)
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords,dads)
        return
    
    @staticmethod
    def theChainFilt(self,filters,dads,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        chain_cm = self.chains_CM(coords)
        
        if frame == self.first_frame:
            self.time_zero=time
        key = self.get_timekey(time,self.time_zero)
        
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, coords, dads)
        
        return 
    @staticmethod
    def box_mean(self,box):
        frame = self.current_frame
        box+=self.get_box(frame)
        return
    
    @staticmethod
    def box_var(self,box_var,box_mean_squared):
        frame = self.current_frame
        box_var += self.get_box(frame)**2 - box_mean_squared
        return
    
    @staticmethod
    def vector_correlations(self,vec_t,filt_t,bk0,bk1,correlation):
            
        frame = self.current_frame
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
              
                try:
                    costh = costh__parallelkernel(v0,v1)
                    correlation[kf][k].append( costh )
                except ZeroDivisionError:
                    pass
        return
                
    @staticmethod
    def segmental_dipole_moment(self,filters,dads,ids1,ids2,
                                segment_args,dipoles_t,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_whole_coords(frame)
        
        box = self.get_box(frame)
        time = self.get_time(frame)
        

        if frame == self.first_frame:
            self.time_zero=time
        
        key = self.get_timekey(time,self.time_zero)
        
        n = segment_args.shape[0]
        
        dipoles = np.empty((n,3),dtype=float)
        pc = self.partial_charge.reshape((self.partial_charge.shape[0],1))
        
        numba_dipoles(pc,coords,segment_args,dipoles)
        
        dipoles_t[key] = dipoles  
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords,dads)
        
        return
    @staticmethod
    def chain_dipole_moment(self,filters,dads,dipoles_t,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_whole_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        
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
                            chain_cm, coords, dads)
        
        return
    
    @staticmethod
    def mass_density_profile(self,nbins,bins,
                                  rho,mass_pol):
        frame = self.current_frame
        
        coords,box,d,cs = self.get_frame_basics(frame)
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += np.sum(mass_pol[fin_bin])/vol_bin
        return
    
    @staticmethod
    def mass_density_profile__pertype(self,nbins,bins,
                                  rho,rho_per_atom_type,ftt,mass_pol):
        
        frame = self.current_frame
        coords,box,d,cs = self.get_frame_basics(frame)
       
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d, bins[i], bins[i+1])
            rho[i] += numba_sum(mass_pol[fin_bin])/vol_bin
  
            for t in rho_per_atom_type.keys():
                ft = np.logical_and( fin_bin,ftt[t])
                rho_per_atom_type[t][i] += numba_sum(mass_pol[ft])/vol_bin
        return
    

    
    @staticmethod
    def mass_density_profile_flux(self,nbins,bins,
                                  rho,mass_pol,rho_mean_sq):
        frame = self.current_frame
        coords,box,d,cs = self.get_frame_basics(frame)
      
        #2) Caclulate profile
        
        for i in range(nbins):     
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += (np.sum(mass_pol[fin_bin])/vol_bin)**2-rho_mean_sq[i]
        return
    
    @staticmethod
    def number_density_profile__2side(self,nbins,bins,
                                  rho_up,rho_down):
        frame = self.current_frame
        coords = self.translated_coords(frame)
        box = self.get_box(frame)
        cs = self.get_particle_cm(coords)
        
        dfun = getattr(Distance_Functions,self.conftype +'__2side')
        d = dfun(self,coords[self.polymer_filt],cs) 
         # needed because in volfun the volume of each bin is multiplied by 2
        #2) Caclulate profile
        
        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])*0.5
            fin_bin_up =   filt_uplow(d,bins[i],bins[i+1])
            fin_bin_down = filt_uplow(d,-bins[i+1],-bins[i])
            rho_up[i] += np.count_nonzero(fin_bin_up)/vol_bin
            rho_down[i] += np.count_nonzero(fin_bin_down)/vol_bin
            
        return
    
    
    @staticmethod
    def mass_density_profile__2side(self,nbins,bins,
                                  rho_up,rho_down,mass_pol):
        frame = self.current_frame
        coords = self.translated_coords(frame)
        box = self.get_box(frame)
        cs = self.get_particle_cm(coords)
        
        dfun = getattr(Distance_Functions,self.conftype +'__2side')
        d = dfun(self,coords[self.polymer_filt],cs) 
         # needed because in volfun the volume of each bin is multiplied by 2
        #2) Caclulate profile
        
        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])*0.5
            fin_bin_up =   filt_uplow(d,bins[i],bins[i+1])
            fin_bin_down = filt_uplow(d,-bins[i+1],-bins[i])
            rho_up[i] += np.sum(mass_pol[fin_bin_up])/vol_bin
            rho_down[i] += np.sum(mass_pol[fin_bin_down])/vol_bin
            
        return
    
    @staticmethod
    def number_density_profile__pertype(self,nbins,bins,
                                  rho,rho_per_atom_type,ftt):
        frame = self.current_frame
        coords,box,d,cs = self.get_frame_basics(frame)
       
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d, bins[i], bins[i+1])
            rho[i] += np.count_nonzero(fin_bin)/vol_bin
  
            for t in rho_per_atom_type.keys():
                ft = np.logical_and( fin_bin,ftt[t])
                rho_per_atom_type[t][i] += np.count_nonzero(ft)/vol_bin
    
    @staticmethod
    def number_density_profile(self,nbins,bins,rho):
        frame = self.current_frame
        coords,box,d,cs = self.get_frame_basics(frame)
        
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += np.count_nonzero(fin_bin)/vol_bin
        return
  
    @staticmethod
    def massnumber_density_profile__conformations(self,dads,dlayers,dens,stats):                
        frame = self.current_frame
        coords = self.get_whole_coords(frame)
        box = self.get_box(frame)
        #1) ads_chains, trains,tails,loops,bridges
        ads_chains, args_train, args_tail, args_loop, args_bridge = self.conformations(dads,coords)
        
        #check_occurances(np.concatenate((args_train,args_tail,args_bridge,args_loop)))

        coreFunctions.conformation_dens(self, dlayers, dens,
                                           args_train, args_tail,
                                           args_loop, args_bridge)
        
        coreFunctions.conformation_stats(stats,ads_chains, args_train, args_tail, 
                             args_loop, args_bridge)
        return
    
    @staticmethod
    def get_args_free(self,args_train,args_tail,args_loop,args_bridge):
        all_not_free = np.concatenate((args_train,args_tail,args_loop,args_bridge))
        f = np.logical_not(np.isin(self.polymer_ids, all_not_free))
        args_free = self.polymer_ids[f]
        return args_free
    
    @staticmethod
    def conformation_dens(self, dlayers,dens,
                             args_train, args_tail, 
                             args_loop, args_bridge):
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        d = self.get_distance_from_particle(coords)
        
        args_free = coreFunctions.get_args_free(self,args_train,
                        args_tail,args_loop,args_bridge)
        
        d_tail = d[args_tail]
        d_loop = d[args_loop]          
        d_bridge = d[args_bridge]
        d_free = d[args_free]
        d_train = d[args_train]
        for l,dl in enumerate(dlayers):
            args_tl = args_tail[filt_uplow(d_tail, dl[0], dl[1])]
            args_lp = args_loop[filt_uplow(d_loop, dl[0], dl[1])]
            args_br =  args_bridge[filt_uplow(d_bridge, dl[0], dl[1])]
            args_fr = args_free[filt_uplow(d_free,dl[0],dl[1])]
            args_tr = args_train[filt_uplow(d_train,dl[0],dl[1])]
            
            vol_bin=self.volfun(self,dl[0],dl[1])
            
            dens['ntrain'][l] += args_tr.shape[0]/vol_bin
            dens['ntail'][l] += args_tl.shape[0]/vol_bin
            dens['nloop'][l] += args_lp.shape[0]/vol_bin
            dens['nbridge'][l] += args_br.shape[0]/vol_bin
            dens['nfree'][l] += args_fr.shape[0]/vol_bin
            
            dens['mtrain'][l] += numba_sum(self.atom_mass[args_tr])/vol_bin
            dens['mtail'][l] += numba_sum(self.atom_mass[args_tl])/vol_bin
            dens['mloop'][l] += numba_sum(self.atom_mass[args_lp])/vol_bin
            dens['mbridge'][l] += numba_sum(self.atom_mass[args_br])/vol_bin
            dens['mfree'][l] += numba_sum(self.atom_mass[args_fr])/vol_bin
        return    

    @staticmethod
    def conformation_stats(stats,ads_chains, args_train, args_tail, 
                             args_loop, args_bridge):
        stats['train'] += args_train.shape[0]
        stats['adschains'] += ads_chains.shape[0] 
        stats['looptailbridge'] += (args_loop.shape[0]+args_tail.shape[0]+args_bridge.shape[0])
        stats['tail'] += args_tail.shape[0]
        stats['loop'] += args_loop.shape[0]
        stats['bridge'] += args_bridge.shape[0]
        return 
    

    
    @staticmethod
    def P2(self,ids1,ids2,dlayers,costh,costh_unv):
        frame = self.current_frame
        #1) coords
        coords = self.get_coords(frame)
        box =self.get_box(frame)

        #2) calc_particle_cm
        cs = self.get_particle_cm(coords)
        
        r1 = coords[ids1]; r2 = coords[ids2]
        
        rm = 0.5*(r1+r2)
        
        d = self.get_distance_from_particle(rm)
        uv = self.unit_vectorFun(self,rm,cs)
        
        costhsquare__kernel(costh,r2-r1,uv)
        
        for i,dl in enumerate(dlayers):
            filt = filt_uplow(d, dl[0], dl[1])
            costh_unv[i].extend(costh[filt])
        
        return   
    @staticmethod
    def P2_conformation(self,ids1,ids2,dlayers,
                        dads,conformation,costh,costh_unv):
        frame = self.current_frame
        #1) coords
        coords = self.get_coords(frame)
        box =self.get_box(frame)

        #2) calc_particle_cm
        cs = self.get_particle_cm(coords)
        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations(dads,coords)
        
        args_free = coreFunctions.get_args_free(self,args_train,
                        args_tail,args_loop,args_bridge)
        
        #get conformation args
        args = locals()['args_'+conformation]
        filt_ids = Filters.filt_bothEndsIn(ids1,ids2,args)
        
        r1 = coords[ids1[filt_ids]]; r2 = coords[ids2[filt_ids]]
        costh = costh[filt_ids]
        rm = 0.5*(r1+r2)
        
        d = self.get_distance_from_particle(rm)
        uv = self.unit_vectorFun(self,rm,cs)
        
        costhsquare__kernel(costh,r2-r1,uv)
        
        for i,dl in enumerate(dlayers):
            filt = filt_uplow(d, dl[0], dl[1])
            costh_unv[i].extend(costh[filt])
        
        return  
    @staticmethod
    def chain_characteristics(self,chain_args,dlayers,chars):
        #1) translate the system
        frame = self.current_frame
        coords = self.get_whole_coords(frame)
        box = self.get_box(frame)
        cs = self.get_particle_cm(coords)
        
        
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
            d =1e16 
            for L in self.box_add(box):
                d = np.minimum(d,self.dfun(self,ch_cm.reshape(1,3),cs+L))
            for i,dl in enumerate(dlayers):
                if dl[0]< d[0] <=dl[1]:
                    for char in chars:
                        chars[char][i].append(local_dict[char])
                    break
        return
    
    @staticmethod
    def dihedrals_t(self,
                dih_ids, ids1, ids2, filters, dads, dihedrals_t, filt_per_t):
        
        t0 = perf_counter()
        frame = self.current_frame
        coords = self.get_whole_coords(frame)
        cm = self.get_particle_cm(coords)
        
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        dih_val = np.empty(dih_ids.shape[0],dtype=float) # alloc 
        dihedral_values_kernel(dih_ids,coords,dih_val)

        if frame == self.first_frame :
            self.time_zero = time
        key = self.get_timekey(time, self.time_zero)
        dihedrals_t[key] = dih_val
        
        del dih_val #deallocating for safety
        tm = perf_counter()
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords,dads)
        tf = perf_counter()
        if frame ==1:
            logger.info('Dihedrals_as_t: Estimate time consuption --> Main: {:2.1f} %, Filters: {:2.1f} %'.format((tm-t0)*100/(tf-t0),(tf-tm)*100/(tf-t0)))
        return

    @staticmethod
    def vects_t(self,ids1,ids2,filters,dads,vec_t,filt_per_t):
        frame = self.current_frame
        coords = self.get_coords(frame)
        
        time = self.get_time(frame)
        
        
        vec = coords[ids2,:] - coords[ids1,:]
        
        if frame == self.first_frame:
            self.time_zero = time
            
        key = self.get_timekey(time,self.time_zero)
        
        vec_t[key] = vec
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords,dads)
        return     
    
    @staticmethod
    def confs_t(self,dads,confs_t):
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations(dads,coords)
        x = dict()
        ntot = args_train.shape[0] + args_tail.shape[0] +\
               args_loop.shape[0] + args_bridge.shape[0]
        for k in ['train','tail','loop','bridge']:
            args = locals()['args_'+k]
            x['x_'+k] = args.shape[0]/ntot
            x['n_'+k] = args.shape[0]
        x['x_ads_chains'] = ads_chains.shape[0]/len(self.chain_args)
        x['n_ads_chains'] = ads_chains.shape[0]
        
        if frame == self.first_frame:
            self.time_zero = time
            
        key = self.get_timekey(time,self.time_zero)
        
        confs_t[key] = x
        
        return

    @staticmethod
    def confs_t__length(self,dads,confs_t):
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations(dads,coords)
        x = dict()
        for args,lab in zip([args_train, args_tail, args_loop, args_bridge],
                        ['l_train','l_tail','l_loop','l_bridge']):
            connected_chunks = self.connected_chunks(args)
            lengths = [self.length_of_connected_chunks(list(ch),coords)
                       for ch in connected_chunks ]
            
            m = np.mean(lengths)
            std = np.std(lengths)
            x[lab] = m
            x[lab+'(std)'] = std
            x[lab+'_lenghts'] = lengths 
        if frame == self.first_frame:
            self.time_zero = time
            
        key = self.get_timekey(time,self.time_zero)
        
        confs_t[key] = x
        
        return  
    
    @staticmethod
    def confs_t__size(self,dads,confs_t):
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations(dads,coords)
        x = dict()
        for args,lab in zip([args_train, args_tail, args_loop, args_bridge],
                        ['s_train','s_tail','s_loop','s_bridge']):
            connected_chunks = self.connected_chunks(args)
            sizes = [s.__len__() for s in connected_chunks]
            size_m = np.mean(sizes)
            size_std = np.std(sizes)
            x[lab] = size_m 
            x[lab+'(std)'] = size_std
            x[lab+'_sizes'] = sizes
        if frame == self.first_frame:
            self.time_zero = time
            
        key = self.get_timekey(time,self.time_zero)
        
        confs_t[key] = x
        
        return    
    
    
    @staticmethod
    def confs_t__perchain(self,dads,confs_t):
        frame = self.current_frame
        coords,box,d,cs = self.get_whole_frame_basics(frame)
        time = self.get_time(frame)
        
        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations(dads,coords)
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
    def chainCM_t(self,filters,dads,vec_t,filt_per_t):
        
        frame = self.current_frame

        coords = self.get_coords(frame)

        
        
        box = self.get_box(frame)
        time = self.get_time(frame)
        
        chain_cm = self.chains_CM(coords)

        
        if frame == self.first_frame:
            self.time_zero=time
        key = self.get_timekey(time,self.time_zero)
        
        cm = CM(coords[self.particle_filt],self.particle_mass)
        vec_t[key] =  chain_cm - cm
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, coords, dads)
        
        return 
    
    @staticmethod
    def segCM_t(self,filters,dads,
                  ids1,ids2,segment_ids,
                  vec_t,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        time = self.get_time(frame)
        #coords_whole = self.get_whole_coords(frame)  
        
        seg_cm = self.segs_CM(coords,segment_ids)
        
        if frame == self.first_frame:
            self.time_zero=time
        key = self.get_timekey(time,self.time_zero)
        
        cm = CM(coords[self.particle_filt],self.particle_mass)
        vec_t[key] =  seg_cm -cm
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            ids1,ids2,coords,dads)
        
        return 
    
    @staticmethod
    def gofr(self,bins,gofr):
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        
        n = coords.shape[0]
        pd = np.empty(int(n*(n-1)/2),dtype=float)

        pair_dists(coords,box,pd)
        pd = pd[pd<=bins.max()]
        numba_bin_count(pd,bins,gofr)
        
        return
    
    @staticmethod
    def gofr_type(self,fty,bins,gofr):
        frame = self.current_frame
        coords = self.get_coords(frame)[fty]
        box = self.get_box(frame)
        
        n = coords.shape[0]
        pd = np.empty(int(n*(n-1)/2),dtype=float)
 
        pair_dists(coords,box,pd)
        pd = pd[pd<=bins.max()]
        numba_bin_count(pd,bins,gofr)
        return
    
    @staticmethod
    def gofr_type_to_all(self,fty,bins,gofr):
        frame = self.current_frame
        coords = self.get_coords(frame)
        
        coords_ty = coords[fty]
        box = self.get_box(frame)
        
        n = coords.shape[0]
        nty = coords_ty.shape[0]
        pd = np.empty(int(n*nty),dtype=float)
        
        pair_dists_general(coords,coords_ty,box,pd)
        pd = pd[pd<=bins.max()]
        numba_bin_count(pd,bins,gofr)

        return
    
    @staticmethod
    def gofr_type_to_type(self,fty1,fty2,bins,gofr):
        frame = self.current_frame
        coords = self.get_coords(frame)
        
        coords_ty1 = coords[fty1]
        coords_ty2 = coords[fty2]
        box = self.get_box(frame)
        
        nty1 = coords_ty1.shape[0]
        nty2 = coords_ty2.shape[0]
        pd = np.empty(int(nty1*nty2),dtype=float)
        
        pair_dists_general(coords_ty1,coords_ty2,box,pd)
        pd = pd[pd<=bins.max()]
        numba_bin_count(pd,bins,gofr)

        return

@jit(nopython=True,fastmath=True)
def fill_property(prop,nv,i,j,value,mi,block_average):
    
    idx = j-i
    if block_average:
        try:
            prop[idx] +=  value/mi
            nv[idx] += 1.0
        except:
            pass
    else:
        prop[idx] +=  value
        nv[idx] += mi
    
    return


@jit(nopython=True,fastmath=True,parallel=True)
def Kinetics_kernel(func,func_args,
                    Prop,nv,xt,wt=None,
                    block_average=False):
    
    n = xt.shape[0]
    
    for i in range(n):
        for j in prange(i,n):
            args = func_args(i,j,xt,None,wt)
            
            value,mi = func(*args)
            fill_property(Prop,nv,i,j,value,mi,block_average)
        
    for i in prange(n):  
        Prop[i] /= nv[i]
    return 

@jit(nopython=True,fastmath=True)
def Kinetics_inner__kernel(x0,xt):
    value = 0.0 ; m = 0.0
    for i in range(x0.shape[0]):
        if x0[i]:
            if xt[i]:
                value += 1.0
            m += 1.0
    return value,m

@jit(nopython=True,fastmath=True)
def Kinetics_inner_weighted__kernel(x0,xt,w0):
    value = 0.0 ; m = 0.0
    for i in range(x0.shape[0]):
        if x0[i]:
            wi = w0[i]
            m += wi
            
            if xt[i]:
                value += wi
            
    return value,m

@jit(nopython=True,fastmath=True)
def get_zero__args(i,xt,ft,wt):
    return (xt[i],)

@jit(nopython=True,fastmath=True)
def get_zero_filt__args(i,xt,ft,wt):
    return (xt[i],ft[i])

@jit(nopython=True,fastmath=True)
def get_zero_weighted__args(i,xt,ft,wt):
    return (xt[i],  wt[i])

@jit(nopython=True,fastmath=True)
def get_zero_filt_weighted__args(i,xt,ft,wt):
    return (xt[i],ft[i],wt[i])

@jit(nopython = True,fastmath=True)
def mean__kernel(ifunc,x):
    mean =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        mean += ifunc(x[i])
    mi = float(x.shape[0])
    return mean, mi

@jit(nopython = True,fastmath=True)
def mean_weighted__kernel(ifunc,x,w):
    mean =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        wi = w[i]
        mean += wi*ifunc(x[i])
        mi += wi
    return mean,mi

@jit(nopython = True,fastmath=True)
def mean_filt__kernel(ifunc,x,f):
    mean =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        if f[i]:
            mean+=ifunc(x[i])
            mi+=1.0
    return mean, mi

@jit(nopython = True,fastmath=True)
def mean_filt_weighted__kernel(ifunc,x,f,w):
    mean =0 ; mi=0
    for i in range(x.shape[0]):
        if f[i]:
            wi = w[i]
            mean+=wi*ifunc(x[i])
            mi+=wi
    return mean, mi


@jit(nopython = True,fastmath=True)
def secmoment__kernel(ifunc,x):
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        xi = ifunc(x[i])
        sec += xi*xi
    mi = float(x.shape[0])
    return sec, mi

@jit(nopython = True,fastmath=True)
def secmoment_weighted__kernel(ifunc,x,w):
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        wi = w[i]
        xi = ifunc(x[i])
        sec += wi*xi*xi
        mi += wi
    return sec,mi

@jit(nopython = True,fastmath=True)
def secmoment_filt__kernel(ifunc,x,f):
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        if f[i]:
            xi = ifunc(x[i])
            sec+=xi*xi
            mi+=1.0
    return sec, mi

@jit(nopython = True,fastmath=True)
def secmoment_filt_weighted__kernel(ifunc,x,f,w):
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        if f[i]:
            wi = w[i]
            xi = ifunc(xi[i])
            sec+=wi*xi*xi
            mi+=w[i]
    return sec, mi


@jit(nopython=True,fastmath=True,parallel=True)
def TACF_kernel(func, func_args, inner_func,
              mean_func, secmoment_func, func_args_zero, inner_func_zero,
              Prop,nv,
              mu_val,mu_num,secmom_val,secmom_num,
              xt, ft=None, wt=None,
              block_average=False):
    
    n= xt.shape[0]
    
    for i in range(n):

        args_zero = func_args_zero(i,xt,ft,wt)
        
        mu_val[i],mu_num[i] = mean_func(inner_func_zero,*args_zero)
        
        secmom_val[i],secmom_num[i] = secmoment_func(inner_func_zero,*args_zero)
        
        for j in prange(i,n):
                
            args = func_args(i,j,xt,ft,wt)

            value,mi = func(inner_func,*args)
            
            fill_property(Prop,nv,i,j,value,mi,block_average)
            
    if block_average:
        for i in range(n):
            
            mui = mu_val[i]/mu_num[i]
            seci = secmom_val[i]/secmom_num[i] 
            
            mui_square = mui*mui  
            vari = seci - mui_square
            
            Prop[i] = (Prop[i]/nv[i] - mui_square)/vari

        return
    else:
        mu =0 ;nmu =0
        sec = 0;nsec =0
        for i in prange(n):
            mu+=mu_val[i]  ; nmu+=mu_num[i]
            sec+= secmom_val[i] ; nsec += secmom_num[i]
        mu/=nmu 
        sec/=nsec
        
        mu_sq = mu*mu
        var = sec-mu_sq
        
        for i in prange(n):
            Prop[i] = (Prop[i]/nv[i] - mu_sq)/var
    
        return 

@jit(nopython=True,fastmath=True,parallel=True)
def DynamicProperty_kernel(func,func_args,inner_func,
              Prop,nv,xt,ft=None,wt=None,
              block_average=False):
    
    n = xt.shape[0]

    for i in range(n):
        for j in prange(i,n):
            
            args = func_args(i,j,xt,ft,wt)
            
            value,mi = func(inner_func,*args)
            
            fill_property(Prop,nv,i,j,value,mi,block_average)
        
    for i in prange(n):   
        Prop[i] /= nv[i]
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
def dynprop__kernel(inner_kernel,r1,r2):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        inner = inner_kernel(r1[i],r2[i])
        tot+=inner
        mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_simple__kernel(inner_kernel,r1,r2,ft0):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            inner = inner_kernel(r1[i],r2[i])
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_strict__kernel(inner_kernel,r1,r2,ft0,fte):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and fte[i]:
            inner = inner_kernel(r1[i],r2[i])
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_change__kernel(inner_kernel,r1,r2,ft0,fte):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and not fte[i]:
            inner = inner_kernel(r1[i],r2[i])
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_weighted__kernel(inner_kernel,r1,r2,w):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        inner = inner_kernel(r1[i],r2[i])
        tot+=w[i]*inner
        mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_simple_weighted__kernel(inner_kernel,r1,r2,ft0,w):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            inner = inner_kernel(r1[i],r2[i])
            tot+=w[i]*inner
            mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_strict_weighted__kernel(inner_kernel,r1,r2,ft0,fte,w):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and fte[i]:
            inner = inner_kernel(r1[i],r2[i])
            tot+=w[i]*inner
            mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_change_weighted__kernel(inner_kernel,r1,r2,ft0,fte,w):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and not fte[i]:
            inner = inner_kernel(r1[i],r2[i])
            tot+=w[i]*inner
            mi+=w[i]
    return tot,mi




@jit(nopython=True,parallel=True)
def numba_bin_count(d,bins,counter):
    for j in prange(bins.shape[0]-1):
        for i in range(d.shape[0]):
            if bins[j]<d[i] and d[i] <=bins[j+1]:
                counter[j] +=1
    return

@jit(nopython=True,fastmath=True)
def fcos_kernel(x):
    return np.cos(x)

@jit(nopython=True,fastmath=True)
def fsin_kernel(x):
    return np.sin(x)

@jit(nopython=True,fastmath=True)
def cosCorrelation_kernel(x1,x2):
    
    c1= np.cos(x1)
    c2 = np.cos(x2)
    return c1*c2

@jit(nopython=True,fastmath=True)
def sinCorrelation_kernel(x1,x2):
    
    s1= np.sin(x1)
    s2 = np.sin(x2)
    return s1*s2



@jit(nopython=True,fastmath=True)
def cos2th_kernel(r1,r2):
    costh = costh_kernel(r1,r2)
    return costh*costh

@jit(nopython=True,fastmath=True)
def costh_kernel(r1,r2):
    costh=0 ; rn1 =0 ;rn2=0
    n = r1.shape[0]
    for j in range(n):
        costh += r1[j]*r2[j]
        rn1 += r1[j]*r1[j]
        rn2 += r2[j]*r2[j]
    rn1 = rn1**0.5
    rn2 = rn2**0.5
    costh/=rn1*rn2
    return costh

@jit(nopython=True,fastmath=True)
def norm_square_kernel(r1,r2):
    
    nm = 0
    for i in range(r1.shape[0]):
        ri = r2[i] - r1[i]
        nm+= ri*ri
    return nm

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
def dihedral_values_kernel(dih_ids,coords,dih_val):
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
def numba_coordination(coords1,coords2,box,maxdist,coordination):
    n1 = coords1.shape[0]
    n2 = coords2.shape[0] 
    for i in prange(n1):
        rel_coords = coords1[i] - coords2
        rc = minimum_image(rel_coords,box)
        dist = np.sqrt(np.sum(rc*rc,axis=1))
        for j in range(n2):
            if dist[j]<maxdist:
                coordination[i]+=1
                
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

@jit(nopython=True,fastmath=True)
def implement_pbc(coords,boxsize):
    cn = coords%boxsize
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

@jit(nopython=True,fastmath=True,parallel=True)
def numba_elementwise_minimum(x1,x2):
    xmin = np.empty_like(x1)
    for i in prange(x1.shape[0]):
        if x1[i]<x2[i]:
            xmin[i] = x1[i]
        else:
            xmin[i] = x2[i]
    return xmin

