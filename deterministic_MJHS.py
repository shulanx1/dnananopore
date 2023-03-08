# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:51:51 2021

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
    (rest potential is 0mV)
    
"""


import numpy as np
import scipy as sci

        
def euler(odes, start, stop, step, initial_values,**kwargs):
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
    
def rk(odes, start, stop, step, initial_values, **kwargs):
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
def vmp_hh(t_c, dt,values, Cm = 0.75, Vna = 60, Vk = -90, Vl = -70, gna_bar = 60, gk_bar = 18, gl_bar = 0.3,area = 6000, area_patch = 5, C_e = 5e-6, R_seal = 75, R_pore = 100000, **kwargs):
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
    n = values[1]
    m = values[2]
    h = values[3]
    area = area/1e8 #cm^2
    area_patch = area_patch/1e8
    g_seal = 1/R_seal/1e3 #mS
    g_pore = 1/R_pore/1e3
    try:
        stim_i = kwargs['stim_i']
        if stim_i:
            i_inj_patch = ip_hh(t_c,dt, values, **kwargs)
        else:
            i_inj_patch = 0
            # print('no i stim given')
    except KeyError:
        i_inj_patch = 0
        # print('no i stim given')

    try:
        stim_i = kwargs['stim_i_step']
        if stim_i:
            i_inj_patch = i_inj_patch + istep_hh(t_c,dt, values,  **kwargs)
        else:
            i_inj_patch = i_inj_patch + 0
            # print('no i stim given')
    except KeyError:
        i_inj_patch = i_inj_patch + 0

    try:
        stim_g = kwargs['stim_g']
        if stim_g:
            Vg = kwargs['Vg']
            i_syn = gp_hh(t_c, dt, values, **kwargs)*(vm-Vg)
        else:
            i_syn = 0
    except KeyError:
        i_syn = 0

    try:
        stim_syn = kwargs['stim_syn']
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

    except KeyError:
        i_syn1 = 0

    try:
        stim_ext = kwargs['stim_ext']
        if stim_ext:
            Ve = kwargs['Ve']
            n_c = int(np.floor(t_c/dt))
            dv_ext = -(Ve[n_c+1]-Ve[n_c])
        else:
            dv_ext = 0
    except KeyError:
        dv_ext = 0

    i_inj = 0
    i_ion = gk_bar*area*n*(vm-Vk) + gna_bar*area*m**3*h*(vm-Vna) + gl_bar*area*(vm-Vl)
    i_ion_p = gk_bar*area_patch*n*(vm-Vk) + gna_bar*area_patch*m**3*h*(vm-Vna) + gl_bar*area_patch*(vm-Vl)
    i_seal = -vp*g_seal
    i_pore = g_pore*(vm-vp)



    A = np.asarray([[Cm*area/dt, C_e/dt], [Cm*area_patch/dt, -Cm*area_patch/dt-C_e/dt]])
    b = np.asarray([i_inj_patch*area-i_seal-i_syn-i_syn1-i_ion-dv_ext*(Cm*area)/dt, -i_inj-i_pore-i_seal])

    f = np.linalg.solve(A, b)

    return f

def np_hh(t_c, dt,values, **kwargs):
    """
    f(t_c, values) = dn/dt

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
    T = 27
    qt = 2.3**((T-16)/10)
    vm = values[0][0]
    n = values[1]
    m = values[2]
    h = values[3]
    if vm ==20:
        vm = vm+0.0001
    alpha_n = qt*0.02*(vm-20)/(1-np.exp(-(vm-20)/9))
    beta_n = -qt*0.002*(vm-20)/(1-np.exp((vm-20)/9))
    tau_n = 1/(alpha_n+beta_n)/qt
    inf_n = alpha_n/(alpha_n+beta_n)
    f = dt*((inf_n-n)/tau_n)
    return f

def mp_hh(t_c, dt,values, **kwargs):
    """
    f(t_c, values) = dm/dt

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
    T = 27
    qt = 3**((T-27)/10)
    vm = values[0][0]
    n = values[1]
    m = values[2]
    h = values[3]
    if vm==-35:
        vm = vm+0.0001
    alpha_m = 0.182*(vm+35)/(1-np.exp(-(vm+35)/9))
    beta_m = -0.124*(vm+35)/(1-np.exp((vm+35)/9))
    tau_m = 1/(alpha_m+beta_m)/qt
    inf_m = alpha_m/(alpha_m+beta_m)
    f = dt*((inf_m-m)/tau_m)
    return f
    
def hp_hh(t_c, dt, values, **kwargs):
    """
    f(t_c, values) = dh/dt

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
    T = 27
    qt = 3**((T-27)/10)
    vm = values[0][0]
    n = values[1]
    m = values[2]
    h = values[3]
    if vm==-50:
        vm = vm+0.0001
    elif vm==-75:
        vm = vm+0.0001
    elif vm==-65:
        vm = vm+0.0001
    ah = -qt*0.024*(vm+50)/(1-np.exp((vm+50)/5))
    bh = qt*0.0091*(vm+75)/(1-np.exp(-(vm+75)/5))
    # ah = -qt*0.024*(vm+65)/(1-np.exp((vm+65)/5))
    # bh = qt*0.02*(vm+65)/(1-np.exp(-(vm+65)/5))
    tau_h = 1/(ah+bh)
    inf_h = 1/(1+np.exp((vm+65)/6.2))
    # inf_h = ah/(ah+bh)
    f = dt*((inf_h-h)/tau_h)
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