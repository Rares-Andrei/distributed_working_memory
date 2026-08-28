import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pandas as pd
import os
import scienceplots
plt.style.use(['science', 'nature'])

plot_params = {
    "figure.dpi": "600",
    "axes.labelsize": 20,
    "axes.linewidth": 1.5,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.title_fontsize": 16,
    "legend.fontsize": 13,
    "xtick.major.size": 3.5,
    "xtick.major.width": 1.5,
    "xtick.minor.size": 2.5,
    "xtick.minor.width": 1.5,
    "ytick.major.size": 3.5,
    "ytick.major.width": 1.5,
    "ytick.minor.size": 2.5,
    "ytick.minor.width": 1.5,
}
plt.rcParams.update(plot_params)


@njit(nogil=True)
def F(x, a, theta):
  """
  Population activation function, F-I curve

  Args:
    x     : the population input
    a     : the gain of the function
    theta : the threshold of the function

  Returns:
    f     : the population activation response f(x) for input x
  """

  f = (1 + np.exp(-a * (x - theta)))**-1 - (1 + np.exp(a * theta))**-1

  return f


@njit(nogil=True)
def simulate_wc(W, sim_time, store_time, dt, dt_store,
                a_E = 0.0, a_I = 0.0, theta_E = 0., theta_I = 0., 
                cEE = 0.0, cEI = 0.0, cIE = 0.0, cII=0.0,
                tauE = 0.0, tauI = 0.0, P = 0.0, I = 0.0, noise = 0.00, 
                rE_0 = 0, rI_0=0, nmda_gradient=0,
                SLN=0, w_long_E=0.0, w_long_I=0.0,
                ts_ext_start=0, ts_ext_end=0, I_ext_strengths=0,
                cEE_min=0.0, cEE_max=0.0):


    '''
    Simulates the network numerically using the Euler method. Can account for different modes of homeostasis.
    Returns the firing rates of excitatory and inhibitory populations and an array tracking the evolution of each parameter that can undergo homeostasis.


    Args
    ----
    W : ndarray
        Structural connectivity matrix
    sim_time : float
        Total simulation time in seconds.
    store_time : float
        Total length of data to be stored, in seconds. The simulator will store the last store_time seconds of model activity
    dt : float 
        Simulation time step, in seconds.
    dt_store : float
        Time step to store the data, in seconds. The simulator will store model activity every dt_store seconds.
    rho : float
        Target firing rate. The mean firing rate at which the model should be maintained.
    muE : float [optional]
        Firing threshold of the excitatory population
    muI : float [optional]
        Firing threshold of the inhibitory population
    sigmaE : float [optional]
        Sensitivity of the excitatory population
    sigmaI : float [optional]
        Sensitivity of the inhibitory population
    cEE : float [optional]
        E->E coupling
    cEI : float [optional]
        E->I coupling
    cIE : float [optional]
        E->I coupling
    gE : float [optional
        Scaling parameter for the excitatory inputs unto the excitatory neural mass    
    tauE : float [optional, default = 2.5e-3]
        Time constant of the excitatory population, in seconds.
    tauI : float [optional, default = 2.5e-3]
        Time constant of the inhibitory population, in seconds.
    P : float [optional, default = 0.8]
        Background input to the excitatory population. Default is set at 0.8, like in Neuromatch tutorial
    I : float or ndarray [optional, default = 0.0]
        External input to the excitatory population. Can be a single constant input (float) or an array with dimension [np.floor(sim_time/dt)]
    noise : float [optional, default = 0.0]
        Variance of Gaussian noise added to the external input to each population
    rE_0 : float or ndarray [optional, default = 0]
        Initial value of rE


    Returns
    -------
    rE_store : ndarray
        Array storing the evolution of rE
    rI_store : ndarray
        Array storing the evolution of rI
    parameters_store : ndarray
        Array storing the evolution of each homeostatic parameter, in the following order [gE, cEI, muE]
    '''
    

    # Gets the number of nodes in the network from the connectivity matrix
    n_nodes = W.shape[0]
    cEE_node = cEE_min + (cEE_max - cEE_min) * nmda_gradient  # per-node E->E recurrence, like J_s
    grad_E = cEE_node / np.max(cEE_node)
    W_E = w_long_E * (W * SLN) * grad_E.reshape(n_nodes, 1)   # feedforward-weighted long-range input, targets E
    W_I = w_long_I * W * (1.0 - SLN)  # feedback-weighted long-range input, targets I
    # Current instantaneous state of each population
    rE = rE_0*np.ones(n_nodes)
    rI = rI_0*np.ones(n_nodes)

    # Arrays to store rE and rI
    rE_store = np.zeros((n_nodes, int(np.floor(store_time/dt_store))))
    rI_store = np.zeros((n_nodes, int(np.floor(store_time/dt_store))))
    i_store = 0  # where to store the data

    # If I is a single value, transforms to array
    I = I * np.ones(int(np.floor(sim_time/dt)))

    for st in range(1, int(np.floor(sim_time/dt))+1):

        # Long-range input from other nodes
        long_range_E = W_E @ rE
        long_range_I = W_I @ rE

        # External cue: extra input to selected nodes during a time window
        I_cue = np.zeros(n_nodes)
        t = (st - 1) * dt  # current time in seconds
        for w in range(len(ts_ext_start)):
            if ts_ext_start[w] <= t < ts_ext_end[w]:
                I_cue = I_cue + I_ext_strengths[w]

        # Integrates equations using the Euler method
        drE = 1/tauE * (-rE + F(cEE_node * rE - cEI*rI + long_range_E + P + I[st-1] + I_cue + noise*np.random.randn(n_nodes), a_E, theta_E))
        drI = 1/tauI * (-rI + F(cIE * rE - cII * rI + long_range_I + noise*np.random.randn(n_nodes), a_I, theta_I))

        rE = rE + dt*drE
        rI = rI + dt*drI

        # Stores the last store_time seconds of activity every dt_store seconds
        if st >= ((sim_time-store_time)/dt) and st % (int(dt_store/dt)) == 0:
            rE_store[:, i_store] = rE
            rI_store[:, i_store] = rI
            i_store += 1

    return rE_store, rI_store
