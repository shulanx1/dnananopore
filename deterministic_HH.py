# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:53:28 2022

@author: xiao208
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:54:28 2019

@author: dalan

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
    area = 1000um^2
    area_patch = 1um^2
    (rest potential is 0mV)
    
"""


import numpy as np
import scipy as sci

        
def euler(odes, start, stop, step, initial_values,kwargs):
    """
    solve a 1 order ODE with euler method. 
    For dy/dt = f(y, t), solve as y[n+1] = y[n] + h*f(t_n, y_n)

    Parameters
    ----------
    odes : list of python function
        ODE(interval, value) = f(t, y), return f
    start : numerical
        Time point of start (in ms)
    stop : numerical
        Time point of stop (in ms)
    step : numerical
        interval size (in ms)
    initial_values : list of numerical values
        the initial value of y (y_0)

    Returns: y, t; lists (note: y will be list of lists)
    -------
    """
    y = [initial_values]
    N = len(initial_values) # number of orders
    if len(odes) != N:
        raise ValueError('The %d ODEs doesn\'t match with %d initial values' % (len(odes),len(initial_values)))
    t = [start]
    dt = step
    for t_c in np.arange(start, stop, step):
        y_c = np.asarray(y[-1])
        y_n = []
        for ode, i in zip(odes, range(N)):
            k = ode(t_c, dt ,y_c, **kwargs)
            y_n.append(y_c[i] + k)
        t_n = t_c + dt
        y.append(y_n)
        t.append(t_n)
    y_transpose = []
    for i in range(N):
        y_transpose.append([x[i] for x in y]) 
    return y_transpose, t 
    
def rk(odes, start, stop, step, initial_values,kwargs):
    """
    solve a 1 order ODE with euler method. 
    For dy/dt = f(y, t), solve with 4 order RK method

    Parameters
    ----------
    odes : list of python functions
        ODE(interval, value) = f(t, y), return f
    start : numerical
        Time point of start (in ms)
    stop : numerical
        Time point of stop (in ms)
    step : numerical
        interval size (in ms)
    initial_values : list
        the initial value of y (y_0)

    Returns: y, t; lists
    -------
    """
    y = [initial_values]
    t = [start]
    # k_all = []
    N = len(initial_values)
    if len(odes) != N:
        raise ValueError('The %d ODEs doesn\'t match with %d initial values' % (len(odes),len(initial_values)))
    dt = step
    for t_c in np.arange(start, stop, step):
        y_c = np.asarray(y[-1]) # current value of y

        k1 = []
        for ode, i in zip(odes, range(N)):
            k1.append(ode(t_c, dt, y_c, **kwargs))

        k2 = []
        for ode, i in zip(odes, range(N)):
            k2.append(ode(t_c+dt/2, dt,y_c + np.asarray(k1)/2, **kwargs))


        k3 = []
        for ode, i in zip(odes, range(N)):
            k3.append(ode(t_c+dt/2, dt, y_c + np.asarray(k2)/2, **kwargs))

        k4 = []
        for ode, i in zip(odes, range(N)):
            k4.append(ode(t_c+dt, dt, y_c + np.asarray(k3), **kwargs))

        y_n = y_c + 1/6*((np.asarray(k1) + 2*np.asarray(k2) + 2*np.asarray(k3) + np.asarray(k4)))
        t_n = t_c + dt # next value fo t
        y.append(list(y_n))
        t.append(t_n)
        # k_all.append([k1,k2,k3,k4])

    y_transpose = []

    for i in range(N):
        y_transpose.append([x[i] for x in y]) 
    return y_transpose, t


# hh equations


def vmp_hh(t_c, dt, values, Cm = 1, Vna = 50, Vk = -77, Vl = -54.4, gna_bar = 120, gk_bar = 36, area = 6000, area_patch = 5, gl_bar = 0.3, C_e = 5e-6, R_seal = 75, R_pore = 100000,
           stim_i = False, stim_i_step = False, stim_g = False, stim_syn = False, stim_ext = False, Ve = 0, Vg = 0, R_shunt_patch = 10000, R_seal_patch = 5000, Ra = 20, stim_param = {}):
    """
    f(t_c, values) = dVm/dt

    Parameters
    ----------
    t_c : numerical
        current time point
    values : list [vm, n, m, h]
        list of variables in the f function
    ip : python function
        I_inj = ip(t)
    gp : python function
        i_syn = gp(t)
    Returns
    -------
    f(t_c, values)

    """
    vm = values[0][0]
    vp = values[0][1]
    vw = values[0][2]
    n = values[1]
    m = values[2]
    h = values[3]
    area = area/1e8 #cm^2
    area_patch = area_patch/1e8
    g_seal = 1/R_seal/1e3 #mS
    g_pore = 1/R_pore/1e3
    g_seal_patch = 1/R_seal_patch/1e3 #mS
    g_shunt_patch = 1/R_shunt_patch/1e3
    g_a = 1/Ra/1e3

    if stim_i:
        i_inj_patch = ip_hh(t_c,dt, values, **stim_param)
    else:
        i_inj_patch = 0
        # print('no i stim given')


    if stim_i_step:
        i_inj_patch = i_inj_patch + istep_hh(t_c,dt, values,  **stim_param)
    else:
        i_inj_patch = i_inj_patch + 0
        # print('no i stim given')


    if stim_g:
        i_syn = gp_hh(t_c, dt, values, **stim_param)*(vm-Vg)
    else:
        i_syn = 0



    if stim_syn:
        mg = 1
        E_rev = 0
        A_A = values[4][0]
        B_A = values[4][1]
        A_N = values[4][2]
        B_N = values[4][3]
        mg_block = 1 / (1 + np.exp(0.062 * -(vm)) * (mg / 3.57))
        i_syn1 = (B_A-A_A)*(vm-E_rev) + (B_N-A_N)*mg_block*(vm-E_rev)
    else:
        i_syn1 = 0


    if stim_ext:
        n_c = int(np.floor(t_c/dt))
        dv_ext = -(Ve[n_c+1]-Ve[n_c])
    else:
        dv_ext = 0


    i_inj = 0
    # vw = (i_inj_patch*area+g_a*vm)/(g_seal_patch+g_a)
    # vw = vm
    i_ion = (gk_bar*area)*n**4*(vm-Vk) + (gna_bar*area)*m**3*h*(vm-Vna) + (gl_bar*area)*(vm-Vl)
    i_ion_p = (gk_bar*area_patch)*n**4*((vm-vp)-Vk) + (gna_bar*area_patch)*m**3*h*((vm-vp)-Vna) + (gl_bar*area_patch)*((vm-vp)-Vl)
    i_seal = -vp*g_seal
    i_pore = g_pore*(vm-vp)
    i_seal1 = vw*g_seal_patch
    i_shunt1 = vm*g_shunt_patch
    i_inj11 = (vw-vm)*g_a
    # i_seal1 = 0
    # i_shunt1 = 0
    # i_inj11 = i_inj_patch*area




    # A = np.asarray([[Cm*area/dt, C_e/dt, 0], [(Cm*area_patch+Cm*area)/dt, -Cm*area_patch/dt, 0], [0,0,C_e/dt]])
    # b = np.asarray([i_inj11-i_shunt1-i_pore-i_ion-i_syn-i_syn1-dv_ext*(Cm*area)/dt, -i_inj*area-i_pore-i_seal, i_inj_patch*area-i_inj11-i_seal1])


    A = np.asarray([[Cm*area/dt, C_e/dt, 0], [(Cm*area_patch)/dt, (-Cm*area_patch-C_e)/dt, 0],  [0,0,C_e/dt]])
    b = np.asarray([i_inj11+i_inj+i_seal-i_shunt1-i_ion-i_syn-i_syn1-dv_ext*(Cm*area)/dt, -i_inj*area-i_pore-i_seal, i_inj_patch*area-i_inj11-i_seal1])

    f = np.linalg.solve(A, b)

    return f

def np_hh(t_c,dt, values, **kwargs):
    """
    f(t_c, values) = dn

    Parameters
    ----------
    t_c : numerical
        current time point
    values : list
        [vm, n, m, h]

    Returns
    -------
    f

    """
    vm = values[0][0]
    n = values[1]
    m = values[2]
    h = values[3]
    alpha_n = 0.01*(-55-vm)/(np.exp((-55-vm)/10)-1)
    beta_n = 0.125*np.exp(-(vm+65)/80)
    f = (alpha_n*(1-n) - beta_n*(n))*dt
    return f

def mp_hh(t_c, dt,values, **kwargs):
    """
    f(t_c, values) = dm

    Parameters
    ----------
    t_c : numerical
        current time point
    values : list
        [vm, n, m, h]

    Returns
    -------
    f

    """
    vm = values[0][0]
    n = values[1]
    m = values[2]
    h = values[3]
    alpha_m = 0.1*(-40-vm)/(np.exp((-40-vm)/10)-1)
    beta_m = 4*np.exp(-(vm+65)/18)
    f = (alpha_m*(1-m) - beta_m*(m))*dt
    return f
    
def hp_hh(t_c, dt,values, **kwargs):
    """
    f(t_c, values) = dh

    Parameters
    ----------
    t_c : numerical
        current time point
    values : list
        [vm, n, m, h]

    Returns
    -------
    f

    """
    vm = values[0][0]
    n = values[1]
    m = values[2]
    h = values[3]
    alpha_h = 0.07*np.exp(-(vm+65)/20)
    beta_h = 1/(np.exp((-35-vm)/10)+1)
    f = (alpha_h*(1-h) - beta_h*(h))*dt
    return f

def ip_hh(t_c, dt, values, **kwargs):
    """
    inject current with a specific waveform

    Parameters
    ----------
    tc : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    try:
        stim_wave = kwargs['i_waveform']
    except():
        return 0
    idx = int(np.floor(t_c/dt))
    return stim_wave[idx]

def istep_hh(t_c, dt, values,  **kwargs):
    """
    step current injection
    params needed: stim_i(bool), stim_start, stim_step, stim_amp
    """
    try:
        stim_start = kwargs['stim_start']
    except():
        stim_start = 0
    try:
        stim_step = kwargs['stim_step']
    except():
        stim_step = 0
    try:
        stim_amp = kwargs['stim_amp']
    except():
        stim_amp = 0
    if t_c>=stim_start and t_c<=stim_start+stim_step:
        i = stim_amp
    else:
        i = 0
    return i
        
def gp_hh(t_c, dt, values,  **kwargs):
    """
    Alpha synapses
    params needed: stim_g(bool), stim_start, tau, g_max, Vg

    """
    try:
        stim_start = kwargs['stim_start']
    except():
        stim_start = 0
    try:
        tau = kwargs['tau']
    except():
        tau = 10
    try:
        g_max = kwargs['g_max']
    except():
        g_max = 0
    if t_c>=stim_start:
        g = g_max*(t_c-stim_start)/tau*np.exp(-(t_c-stim_start-tau)/tau)
    else:
        g = 0
    return g
    
def syn_hh(t_c,dt, values, **kwargs):
    """
    AMPA: double exponential with gmax = 500pS (assuming 1000um^2 area: 50 uS/cm^2), tau = 2
    NMDA: double exponential + Mg deblock with gmax = 8000pS (800uS/cm^2)

    """
    syn_kinetic = values[4] # [A_AMPA,B_AMPA, A_NMDA, B_NMDA]
    A_AMPA = syn_kinetic[0]
    B_AMPA = syn_kinetic[1]
    A_NMDA = syn_kinetic[2]
    B_NMDA = syn_kinetic[3]
    try:
        stim_start = kwargs['stim_start_syn']
    except():
        stim_start = 0
    try:
        n_syn = kwargs['n_syn'] # Number of coactivated synapses
    except():
        n_syn = 1

    weight_AMPA = 1*n_syn
    weight_NMDA = 100*n_syn
    tau1_AMPA = 0.01
    tau2_AMPA = 2
    tau1_NMDA = 32
    tau2_NMDA = 50
    tp_AMPA = (tau1_AMPA*tau2_AMPA)/(tau2_AMPA-tau1_AMPA)*np.log(tau2_AMPA/tau1_AMPA)
    tp_NMDA = (tau1_NMDA*tau2_NMDA)/(tau2_NMDA-tau1_NMDA)*np.log(tau2_NMDA/tau1_NMDA)
    factor_AMPA = 1/(-np.exp(-tp_AMPA/tau1_AMPA)+np.exp(-tp_AMPA/tau2_AMPA))
    factor_NMDA = 1/(-np.exp(-tp_NMDA/tau1_NMDA)+np.exp(-tp_NMDA/tau2_NMDA))
    if abs(np.asarray(t_c)-np.asarray(stim_start))<0.01:
        A_AMPA = A_AMPA + weight_AMPA*factor_AMPA
        B_AMPA = B_AMPA + weight_AMPA*factor_AMPA
        A_NMDA = A_NMDA + weight_NMDA*factor_NMDA
        B_NMDA = B_NMDA + weight_NMDA*factor_NMDA
    dA_AMPA = -A_AMPA/tau1_AMPA*dt
    dB_AMPA = -B_AMPA/tau2_AMPA*dt
    dA_NMDA = -A_NMDA/tau1_NMDA*dt
    dB_NMDA = -B_NMDA/tau2_NMDA*dt
    return np.asarray([dA_AMPA,dB_AMPA,dA_NMDA,dB_NMDA])
