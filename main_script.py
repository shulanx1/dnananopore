# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:25:10 2019

@author: Shulan

list of unit and default parameters:
    time: ms
    voltage: mV
    current: uA/cm^2
    capacitance: uf/cm^2
    conducrance: mS/cm^2

    Cm = 1 uF/cm^2
    Vna = 115mV
    Vk = -12mV
    Vl = 10.613mV
    gna_bar = 120mS/cm^2
    gk_bar = 36mS/cm^2
    gl_bar = 0.3mS/cm^2
"""
import sys
sys.path.insert(1, 'E:\Code\DNAnanopore')

import numpy as np
import matplotlib.pyplot as plt
import deterministic_HH
import scipy.io as sio
from scipy.stats import norm
from numpy.random import RandomState

# additional functions
def plot_deterministic(Vnmh, t, Vna = 50, Vk = -77, gna_bar = 120, gk_bar = 36):
    """
    plot Vm and the Na and K current

    Parameters
    ----------
    Vnmh : list of lists
        the y_rk solved with RK method
    t : list
        list of time series
    Vna : TYPE, optional
        DESCRIPTION. The default is 115.
    Vk : TYPE, optional
        DESCRIPTION. The default is -12.
    gna_bar : TYPE, optional
        DESCRIPTION. The default is 120.
    gk_bar : TYPE, optional
        DESCRIPTION. The default is 36.

    Returns
    -------
    None.

    """
    v = np.vstack(np.asarray(Vnmh[0]))
    v = v.T
    dt = t[1]-t[0]
    vm = v[0]
    vp = v[1]
    n = np.asarray(Vnmh[1])
    m = np.asarray(Vnmh[2])
    h = np.asarray(Vnmh[3])
    i_na = gna_bar*m**3*h*(vm-Vna)
    i_k = gk_bar*n*(vm-Vk)
    fig = plt.figure(figsize = [8, 10])
    plt.subplot(3,1,1)
    plt.plot(t, vm)
    plt.xticks([])
    plt.ylabel('Vm (mV)')
    plt.subplot(312)
    plt.plot(t, vp)
    plt.xticks([])
    plt.ylabel('Vp (mV)')
    plt.subplot(3,1,3)
    plt.plot(t, i_na)
    plt.plot(t, i_k)
    print(1/np.mean(gk_bar*n[int(35/dt):int(45/dt)]**4*6000/1e8*1e3))
    print(1/np.mean(gna_bar*m[int(35/dt):int(45/dt)]**3*h[int(35/dt):int(45/dt)]*6000/1e8*1e3))
    plt.legend(['ina','ik'])
    plt.ylabel('i (uA/cm^2)')
    plt.xlabel('t (ms)')

    

    
def plot_stimulation(Vnmh, t, **kwargs):
    vm = np.asarray(Vnmh[0])
    try:
        stim_i = kwargs['stim_i']
        i = []
        for t_c in t:
            i.append(deterministic_HH.ip_hh(t_c, **kwargs))
        plt.figure(figsize = [8, 3])
        plt.plot(t, i)
        plt.xlabel('t (ms)')
        plt.ylabel('stim_i (uA/cm^2)')
    except KeyError:
        pass
    try:
        stil_g = kwargs['stim_g']
        i = []
        g = []
        for t_c, idx in zip(t, range(len(vm))):
            Vg = kwargs['Vg']
            g.append(deterministic_HH.gp_hh(t_c, **kwargs))
            i.append(deterministic_HH.gp_hh(t_c, **kwargs)*(vm[idx]-Vg))
        fig = plt.figure(figsize = [8, 3])
        ax1 = fig.subplots()
        color = 'tab:red'
        ax1.plot(t, i, color = color)
        ax1.set_xlabel('t (ms)')
        ax1.set_ylabel('stim_i (uA/cm^2)', color = color)
        ax1.tick_params(axis = 'y', labelcolor = color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.plot(t, g, color = color)
        ax2.set_ylabel('stim_g (mS/cm^2)')
        ax2.tick_params(axis = 'y', labelcolor = color)
    except KeyError:
        pass




#%% active
i_step_param = {
        'stim_start': 50,
        'stim_step': 50,
        'stim_amp': 10,}
R_seal = np.asarray([25,50,75,100,200,500,1000,1500,2000])
R_pore = np.asarray([50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0, 64000.0, 128000.0])
gna_bar = 120
gk_bar = 36
for nn in range(len(R_seal)):
    Vm_all = []
    Vp_all = []
    Vw_all = []
    dt = 0.01
    
    RMP = np.zeros(R_pore.shape)
    Esteady = np.zeros(R_pore.shape)
    Rm_r = np.zeros(R_pore.shape)
    Esteady_p = np.zeros(R_pore.shape)
    for i in range(len(R_pore)):
        stim_param = {
                'stim_i_step':True,
                'R_pore': R_pore[i],
                'R_seal': R_seal[nn],
                'area': 6000,
                'stim_param': i_step_param,
                }
        y_rk, t = deterministic_HH.euler([deterministic_HH.vmp_hh,
                                            deterministic_HH.np_hh, 
                                            deterministic_HH.mp_hh,
                                            deterministic_HH.hp_hh], 
                                            start = 0, stop = 150, step = dt, initial_values = [[-65.0, -0.0, -65.0], 0, 0, 0], kwargs = stim_param)
        v = np.vstack(np.asarray(y_rk[0]))
        n = np.asarray(y_rk[1])
        m = np.asarray(y_rk[2])
        h = np.asarray(y_rk[3])
        v = v.T
        t = np.asarray(t)
        vm = v[0]
        vp = v[1]
        vw = v[2]
        RMP[i] = np.mean(vm[int(35/dt):int(45/dt)])
        Esteady[i] = np.mean(vw[int(35/dt):int(45/dt)])
        Esteady_p[i] = np.mean(vp[int(35/dt):int(45/dt)])
        Rna = 1/np.mean(gna_bar*m[int(35/dt):int(45/dt)]**3*h[int(35/dt):int(45/dt)]*6000/1e8*1e3)
        Rk = 1/np.mean(gk_bar*n[int(35/dt):int(45/dt)]**4*6000/1e8*1e3)
        Rm_r[i] = 1/(1/55.5+1/Rna+1/Rk)
        Vm_all.append(vm)
        Vp_all.append(vp)
        Vw_all.append(vw)

    
    data = {'Vm':Vm_all,
            'Vp': Vp_all,
            'Vw':Vw_all,
            'R_pore': R_pore,
            'RMP': RMP,
            'Esteady': Esteady,
            'Esteady_p': Esteady_p,
            'Rm': 23.91,
            'Rm_r': Rm_r,
            'Rseal': R_seal[nn],
            't':t}
    sio.savemat(('E:\\Code\\DNAnanopore\\active\\AP_entire_Rseal%d.mat' %R_seal[nn]),data)
#%% passive
i_step_param = {
        'stim_start': 50,
        'stim_step': 50,
        'stim_amp': 10,}
R_seal = np.asarray([25,50,75,100,200,500,1000,1500,2000])
R_pore = np.asarray([50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0, 64000.0, 128000.0])

for n in range(len(R_seal)):
    Vm_all = []
    Vp_all = []
    Vw_all = []
    dt = 0.01
    
    RMP = np.zeros(R_pore.shape)
    Esteady = np.zeros(R_pore.shape)
    Esteady_p = np.zeros(R_pore.shape)
    for i in range(len(R_pore)):
        stim_param = {
                'stim_i_step':True,
                'R_pore': R_pore[i],
                'R_seal': R_seal[n],
                'gna_bar': 0,
                'gk_bar': 0,
                'area': 6000,
                'stim_param': i_step_param,
                }
        y_rk, t = deterministic_HH.euler([deterministic_HH.vmp_hh,
                                            deterministic_HH.np_hh, 
                                            deterministic_HH.mp_hh,
                                            deterministic_HH.hp_hh], 
                                            start = 0, stop = 50, step = dt, initial_values = [[-65.0, -0.0, -65.0], 0, 0, 0], kwargs = stim_param)
        v = np.vstack(np.asarray(y_rk[0]))
        v = v.T
        t = np.asarray(t)
        vm = v[0]
        vp = v[1]
        vw = v[2]
        RMP[i] = np.mean(vm[int(35/dt):int(45/dt)])
        Esteady[i] = np.mean(vw[int(35/dt):int(45/dt)])
        Esteady_p[i] = np.mean(vp[int(35/dt):int(45/dt)])
        Vm_all.append(vm)
        Vp_all.append(vp)
        Vw_all.append(vw)
        
    
    data = {'Vm':Vm_all,
            'Vp': Vp_all,
            'Vw':Vw_all,
            'R_pore': R_pore,
            'RMP': RMP,
            'Esteady': Esteady,
            'Esteady_p': Esteady_p,
            'Rm': 55.5,
            'Rseal': R_seal[n],
            't':t}
    sio.savemat(('E:\\Code\\DNAnanopore\\passive\\passive_entire_Rseal%d.mat' %R_seal[n]),data)
#%%
i_step_param = {
        'stim_start': 50,
        'stim_step': 50,
        'stim_amp': 10,}
R_pore = 1000000.0
stim_param = {
        'stim_i_step':True,
        'R_pore': R_pore,
        'area': 6000,
        'stim_param': i_step_param,
        }
y_rk, t = deterministic_HH.euler([deterministic_HH.vmp_hh,
                                    deterministic_HH.np_hh, 
                                    deterministic_HH.mp_hh,
                                    deterministic_HH.hp_hh], 
                                    start = 0, stop = 150, step = 0.05, initial_values = [[-65.0, -0.0, -65.0], 0, 0, 0], kwargs = stim_param)
plot_deterministic(y_rk, t)





