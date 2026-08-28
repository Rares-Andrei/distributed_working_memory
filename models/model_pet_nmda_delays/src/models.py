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
def calc_J_s(J_MIN, J_MAX, h):
    """
    Calculate the synaptic strength J_s for each brain area.
    """
    return J_MIN + (J_MAX - J_MIN) * h

@njit(nogil=True)
def calc_J_IE(J_0, J_C, J_EI, ZETA, J_s):
    """
    Calculate the synaptic strength J_IE for each brain area.
    """
    return (J_0 - J_s - J_C) / (2 * J_EI * ZETA)

@njit(nogil=True)
def phi_E(A, B, D, I):
    """
    Transfer function for excitatory populations.
    """
    return (A * I - B) / (1 - np.exp(-D * (A * I - B)))

@njit(nogil=True)
def phi_I(R_0, C_1, C_0, G_I, I):
    """
    Transfer function for inhibitory populations.
    """
    return np.maximum(R_0 + (C_1 * I - C_0) / G_I, 0)

@njit(nogil=True)
def calc_dY(A: float, B: float, D: float, R_0: float, C_0: float, C_1: float, 
            G_I: float, J_s: float, J_IE: float, J_EI: float, J_II: float, J_C: float, 
            W_E: np.ndarray, W_I: np.ndarray, I_0A: float, I_0B: float, I_0C: float, 
            GAMMA: float, GAMMA_I: float, SIGMA_A: float, SIGMA_B: float, 
            N: int, Y: np.ndarray, S: np.ndarray, I_long_A: np.ndarray, I_long_B: np.ndarray, I_long_C: np.ndarray,
            dt: float, dt_over_taus: np.ndarray, I_ext: np.ndarray,
            I_mon: np.ndarray) -> np.ndarray:
    """
    Compute the instantaneous derivative of the state matrix Y for an N x 8 neural system.

    This function calculates the differential changes for a system where each of the N neural
    units is represented by 8 state variables (three firing rates, three synaptic gating variables,
    and two noise terms). It integrates the state using the specified time-step and incorporates
    both deterministic and stochastic dynamics.

    Parameters
    ----------
    A : float
        Parameter for the excitatory gain function (phi_E) affecting the response slope.
    B : float
        Parameter for the excitatory gain function (phi_E).
    D : float
        Parameter for the excitatory gain function (phi_E).
    R_0 : float
        Baseline offset for the inhibitory gain function (phi_I).
    C_0 : float
        Parameter for the inhibitory gain function (phi_I) for shaping the response.
    C_1 : float
        Additional parameter for the inhibitory gain function (phi_I).
    G_I : float
        Parameter for the inhibitory gain function (phi_I).
    J_s : float
        Synaptic coupling strength for self (recurrent) excitation.
    J_IE : float
        Synaptic coupling from excitatory to inhibitory populations.
    J_EI : float
        Synaptic coupling from inhibitory to excitatory populations.
    J_II : float
        Synaptic coupling among inhibitory populations.
    J_C : float
        Coupling parameter between excitatory populations A and B.
    W_E : np.ndarray
        Weight matrix for long-range projections between excitatory populations.
    W_I : np.ndarray
        Weight matrix for long-range projections from excitatory to inhibitory populations.
    I_0A : float
        Baseline input current for population A (excitatory).
    I_0B : float
        Baseline input current for population B (excitatory).
    I_0C : float
        Baseline input current for population C (inhibitory).
    GAMMA : float
        Synaptic gating constant for excitatory synapses affecting S_A and S_B.
    GAMMA_I : float
        Synaptic gating constant for inhibitory synapses affecting S_C.
    SIGMA_A : float
        Standard deviation of the noise for population A.
    SIGMA_B : float
        Standard deviation of the noise for population B.
    N : int
        Number of brain nodes; must match the number of rows in Y and I_ext.
    Y : np.ndarray, shape (N, 8)
        Current state matrix where each row consists of:
            - Columns 0-2: Firing rates (r_A, r_B, r_C)
            - Columns 3: Synaptic gating variables (S_C)
            - Columns 4-5: Noise terms (x_A, x_B)
    S: np.ndarray, shape (N, np.floor(t_eval/dt), 2)
    dt : float
        Time step size for integration.
    dt_over_taus : np.ndarray, length 5
        Precomputed time step ratios for various time constants:
            - dt / tau_r, dt / tau_N, dt / tau_G, dt / tau_noise, sqrt(dt / tau_noise)
    I_ext : np.ndarray, shape (N, 3)
        External input for each unit, with columns corresponding to populations A, B, and C.

    Returns
    -------
    np.ndarray, shape (N, 8)
        Differential change for the state matrix Y. The columns correspond to:
            - 0-2: Changes in firing rates (dr_A, dr_B, dr_C)
            - 3-5: Changes in synaptic gating variables (dS_A, dS_B, dS_C)
            - 6-7: Changes in noise variables (dx_A, dx_B)
    
    Notes
    -----
    - The excitatory and inhibitory firing rate updates are computed using the functions phi_E and phi_I,
      defined above.
    - The noise updates incorporate a random component generated using np.random.randn.
    """
    
    dS = np.zeros_like(S)
    
    # Extract variables and precomputed time step ratios
    r_A, r_B, r_C = Y[:, 0], Y[:, 1], Y[:, 2]
    S_C = Y[:, 3]
    S_A, S_B = S[:, 0], S[:, 1]
    x_A, x_B = Y[:, 4], Y[:, 5]
    dt_over_tau_r, dt_over_tau_N, dt_over_tau_G, dt_over_tau_noise, sqrt_dt_over_tau_noise = dt_over_taus

    # Compute the inputs
    I_A = J_s * S_A + J_C * S_B + J_EI * S_C + I_0A + I_long_A + x_A + I_ext[:, 0]
    I_B = J_C * S_A + J_s * S_B + J_EI * S_C + I_0B + I_long_B + x_B + I_ext[:, 1]
    I_C = J_IE * S_A + J_IE * S_B + J_II * S_C + I_0C + I_long_C + I_ext[:, 2]

    I_mon[:, 0] = J_s * S_A + J_C * S_B + J_EI * S_C
    I_mon[:,1] = I_long_A
    I_mon[:,2] = I_long_B

    # Compute the differentials of firing rates
    dr_A = (-r_A + phi_E(A=A, B=B, D=D, I=I_A)) * dt_over_tau_r
    dr_B = (-r_B + phi_E(A=A, B=B, D=D, I=I_B)) * dt_over_tau_r
    dr_C = (-r_C + phi_I(R_0=R_0, C_1=C_1, C_0=C_0, G_I=G_I, I=I_C)) * dt_over_tau_r

    # Compute the differentials of synaptic conductances
    dS_A = -S_A * dt_over_tau_N + GAMMA * (1 - S_A) * r_A * dt
    dS_B = -S_B * dt_over_tau_N + GAMMA * (1 - S_B) * r_B * dt
    dS_C = -S_C * dt_over_tau_G + GAMMA_I * r_C * dt

    # Compute the differentials of noise terms (without instantaneous part)

    # generate one OU-noise draw per unit
    eA = np.random.randn(N)
    eB = np.random.randn(N)

    # base OU update for all regions
    dx_A = -x_A * dt_over_tau_noise + SIGMA_A * sqrt_dt_over_tau_noise * eA
    dx_B = -x_B * dt_over_tau_noise + SIGMA_B * sqrt_dt_over_tau_noise * eB

    # Combine the differentials
    dY = np.zeros_like(Y)
    dY[:, 0] = dr_A
    dY[:, 1] = dr_B
    dY[:, 2] = dr_C
    dY[:, 3] = dS_C
    dY[:, 4] = dx_A
    dY[:, 5] = dx_B
    dS[:, 0] = dS_A
    dS[:, 1] = dS_B
    
    return dY, dS, I_mon

# Function for BOLD signal, not used in this paper
@njit(nogil=True)
def balloon_windkessel_A_B(state_history, dt):
    # Here the state history should come from run_SA_SB rather than run 
    # Calculate BOLD signal generated by synaptic variables S_A and S_B
    sum_state = state_history[:, :, 0] + state_history[:, :, 1]
    print('Sum state shape:', np.shape(sum_state))
    BOLD = np.zeros(np.shape(sum_state)) # This will have shape (time_points, number_of_regions)
    print('BOLD shape:', np.shape(BOLD))
    n = np.shape(sum_state)[1] # The number of regions
    print(n)
    # Initial conditions
    x = 0 * np.ones((1, n))
    f = np.ones((1, n))
    v = np.ones((1, n))
    q = np.ones((1, n))
    
    k = 0.65 # Hz
    ro = 0.34 # Resting oxygen extraction fraction
    tau = 0.98 # s, hemodynamic transit time
    k1 = 3.72 # Dimensionless magnetic field strength-dependent parameter
    k2 = 0.53 # Dimensionless magnetic field strength-dependent parameter
    k3 = 0.53 # Dimensionless magnetic field strength-dependent parameter
    gamma = 0.41 # Hz, rate of flow-dependent elimination
    alpha = 0.32 # Grubb's exponent
    V0 = 0.02 # Resting blood volume fraction
    
    for i in range(1, np.shape(sum_state)[0]): # We iterate over time points
        dx = (sum_state[i - 1, :] - k *x - gamma*(f - 1)) * dt
        df = x*dt
        dv = (f - v**(1/alpha)) * (dt/tau)
        dq = ((f/ro) * ((1 - (1 - ro)**(1/f))) - q*(v**((1/alpha) - 1))) * (dt/tau)
        
        x += dx
        f += df
        v += dv
        q += dq
        
        y = V0 * (k1*(1 - q) + k2*(1 - (q/v)) + k3*(1 - v))
        BOLD[i, :] = y.flatten()
        
    return BOLD

@njit(nogil=True)
def balloon_windkessel_A(state_history, dt):
    # Calculate BOLD signal generated by synaptic variables S_A
    sum_state = state_history[:, :, 3]
    BOLD = np.zeros(np.shape(sum_state)) # This will have shape (time_points, number_of_regions)
    n = np.shape(sum_state)[1] # The number of regions
    # Initial conditions
    x = 0 * np.ones((1, n))
    f = np.ones((1, n))
    v = np.ones((1, n))
    q = np.ones((1, n))
    
    k = 0.65 # Hz
    ro = 0.34 # Resting oxygen extraction fraction
    tau = 0.98 # s, hemodynamic transit time
    k1 = 3.72 # Dimensionless magnetic field strength-dependent parameter
    k2 = 0.53 # Dimensionless magnetic field strength-dependent parameter
    k3 = 0.53 # Dimensionless magnetic field strength-dependent parameter
    gamma = 0.41 # Hz, rate of flow-dependent elimination
    alpha = 0.32 # Grubb's exponent
    V0 = 0.02 # Resting blood volume fraction
    
    for i in range(1, np.shape(sum_state)[0]): # We iterate over time points
        dx = (sum_state[i - 1, :] - k *x - gamma*(f - 1)) * dt
        df = x*dt
        dv = (f - v**(1/alpha)) * (dt/tau)
        dq = ((f/ro) * ((1 - (1 - ro)**(1/f))) - q*(v**((1/alpha) - 1))) * (dt/tau)
        
        x += dx
        f += df
        v += dv
        q += dq
        
        y = V0 * (k1*(1 - q) + k2*(1 - (q/v)) + k3*(1 - v))
        BOLD[i, :] = y.flatten()
        
    return BOLD

@njit(nogil=True)
def balloon_windkessel_B(state_history, dt):
    # Calculate BOLD signal generated by synaptic variables S_B
    sum_state = state_history[:, :, 4]
    BOLD = np.zeros(np.shape(sum_state)) # This will have shape (time_points, number_of_regions)
    n = np.shape(sum_state)[1] # The number of regions
    # Initial conditions
    x = 0 * np.ones((1, n))
    f = np.ones((1, n))
    v = np.ones((1, n))
    q = np.ones((1, n))
    
    k = 0.65 # Hz
    ro = 0.34 # Resting oxygen extraction fraction
    tau = 0.98 # s, hemodynamic transit time
    k1 = 3.72 # Dimensionless magnetic field strength-dependent parameter
    k2 = 0.53 # Dimensionless magnetic field strength-dependent parameter
    k3 = 0.53 # Dimensionless magnetic field strength-dependent parameter
    gamma = 0.41 # Hz, rate of flow-dependent elimination
    alpha = 0.32 # Grubb's exponent
    V0 = 0.02 # Resting blood volume fraction
    
    for i in range(1, np.shape(sum_state)[0]): # We iterate over time points
        dx = (sum_state[i - 1, :] - k *x - gamma*(f - 1)) * dt
        df = x*dt
        dv = (f - v**(1/alpha)) * (dt/tau)
        dq = ((f/ro) * ((1 - (1 - ro)**(1/f))) - q*(v**((1/alpha) - 1))) * (dt/tau)
        
        x += dx
        f += df
        v += dv
        q += dq
        
        y = V0 * (k1*(1 - q) + k2*(1 - (q/v)) + k3*(1 - v))
        BOLD[i, :] = y.flatten()
        
    return BOLD

@njit(nogil=True)
def precompute_dt_over_taus(TAU_R: float, TAU_N: float, TAU_G: float, TAU_NOISE: float, dt: float) -> np.ndarray:
    """
    Precompute time step ratios and the noise scaling factor.

    This function computes the ratios of the time step size `dt` to each of the provided
    time constants (TAU_R, TAU_N, TAU_G, TAU_NOISE) and appends the square root of the 
    ratio corresponding to TAU_NOISE. This function can save time in longer simulations.

    Parameters
    ----------
    TAU_R : float
        Time constant for the firing rate dynamics.
    TAU_N : float
        Time constant for excitatory synaptic gating.
    TAU_G : float
        Time constant for inhibitory synaptic gating.
    TAU_NOISE : float
        Time constant governing noise decay.
    dt : float
        Time step size for numerical integration.

    Returns
    -------
    np.ndarray
        An array of five elements:
            - The first four elements are the ratios: dt/TAU_R, dt/TAU_N, dt/TAU_G, dt/TAU_NOISE.
            - The fifth element is the square root of dt/TAU_NOISE, used for scaling noise.
    """
    taus = np.empty(4, dtype=np.float64)
    taus[0] = float(TAU_R)
    taus[1] = float(TAU_N)
    taus[2] = float(TAU_G)
    taus[3] = float(TAU_NOISE)
    remainders = taus % dt
    assert remainders.all() == 0, f"dt is suggested to be divisible by all time constants: {taus}"
    dt_over_taus = dt / taus
    # Append the square root of the dt ratio for TAU_NOISE
    return np.append(dt_over_taus, np.sqrt(dt_over_taus[-1]))

@njit(nogil=True)
def simulation_run(Y, S, delays, t_eval, dt, dt_over_taus, ts_ext_start, ts_ext_end, I_ext_strengths,
                    A, B, D, R_0, C_0, C_1, G_I, J_s, J_IE, J_C, J_EI, J_II, W_E, W_I, I_0A, I_0B, I_0C,
                    GAMMA, GAMMA_I, SIGMA_A, SIGMA_B, N, lesion_indices, lesion_start_time, I_mon):
    """
    Compute differentials and store state history for the duration of the simulation.
    
    This function iterates through each time step of the simulation using Euler-Maruyama
    integration scheme, starting from the second time step (the first time step is the
    initialization), updates the external current according to the user input, computes
    the differential dY and stores the state history for each time point.
    
    Parameters
    ----------
    Y : np.ndarray, shape (N, 6)
        Current state matrix where each row consists of:
            - Columns 0-2: Firing rates (r_A, r_B, r_C)
            - Columns 3: Synaptic gating variables (S_C)
            - Columns 6-7: Noise terms (x_A, x_B)
    S: np.ndarray, shape (N, np.floor(t_eval/dt), 2):
            - Consists of S_A and S_B variables, stored over time for all regions
    delays: np.ndarray, shape (N, N), contains delays in seconds
    t_eval : float
        Duration of the simulation.
    dt : float
        Time step size for integration.
    dt_over_taus : np.ndarray, length 5
        Precomputed time step ratios for various time constants:
            - dt / tau_r, dt / tau_N, dt / tau_G, dt / tau_noise, sqrt(dt / tau_noise)
    ts_ext_start : np.ndarray, shape (1, x)
        Array of start times for external stimulation, accepts
        arbitrary number of start times.
    ts_ext_end : np.ndarray, shape (1, x)
        Array of end times for external stimulation, accepts
        arbitrary number of end times.
    I_ext_strengths : np.ndarray
        Array (or list) of external input strengths for each period defined in ts_ext_start and ts_ext_end.
    A, B, D : float
        Parameters for the excitatory gain function (phi_E).
    R_0, C_0, C_1, G_I : float
        Parameters for the inhibitory gain function (phi_I).
    J_s, J_IE, J_C, J_EI, J_II : float
        Synaptic coupling strengths between the various neural populations.
    W_E, W_I : np.ndarray
        Weight matrices for E-E and E-I long-range connections.
    I_0A, I_0B, I_0C : float
        Baseline input currents for the respective neural populations.
    GAMMA, GAMMA_I : float
        Synaptic gating constants for excitatory and inhibitory synaptic variables, respectively.
    SIGMA_A, SIGMA_B : float
        Standard deviations for the noise terms.
    N : int
        Number of nodes (should match the number of rows in Y and the first dimension of any N-based inputs).
    lesion_indices : np.ndarray
        Array of indices where the neural units are to be lesioned. For each index, the first
        six state variables are set to zero during the simulation.

    Returns
    -------
    state_history : list of np.ndarray
        A list containing the state matrix Y at each time point (including the initial state).
    Y : np.ndarray
        The final state matrix at the end of the simulation.
    
    """
    # Create a copy of the initial state to avoid modifying the input array.
    Y = Y.copy()
    S = S.copy()
    delays = delays.copy()
    delays = np.floor(delays/dt).astype(np.int64)
    state_history = [Y.copy()]
    n_steps = t_eval.shape[0]  # number of time points
    S_hist = np.zeros((n_steps, N, 2))  # S_hist[k, j, 0]=S_A[j], S_hist[k, j, 1]=S_B[j] at step k
    S_hist[0] = S
    input_history = [I_mon.copy()]
    
    # Loop over the evaluation times (starting from the second timepoint)
    for k, t in enumerate(t_eval[1:], 1):
        # Initialize external current as zero (shape: N x 3)
        I_ext = np.zeros((N, 3))
        if ts_ext_start is not None: # If external input is provided
            # Loop over the defined external input periods and apply input if current time is within a period
            for j, (t_ext_start, t_ext_end) in enumerate(zip(ts_ext_start, ts_ext_end)):
                if t_ext_start <= t < t_ext_end:
                    I_ext += I_ext_strengths[j]
       
        now = k - 1  # current time index; S right now == S_hist[now]
        I_long_A = np.zeros(N)
        I_long_B = np.zeros(N)
        I_long_C = np.zeros(N)
        for i in range(N):  # target
            for j in range(N):  # source
                ksrc = now - delays[i, j]
                if ksrc < 0:
                    ksrc = 0  # To avoid initial period problems
                sa = S_hist[ksrc, j, 0]
                sb = S_hist[ksrc, j, 1]
                I_long_A[i] += W_E[i, j] * sa
                I_long_B[i] += W_E[i, j] * sb
                I_long_C[i] += W_I[i, j] * (sa + sb)
        # Calculate the differentials using the calc_dY function
        dY, dS, I_mon = calc_dY(A=A, B=B, D=D,
                     R_0=R_0, C_0=C_0, C_1=C_1, G_I=G_I,
                     J_s=J_s, J_IE=J_IE, J_EI=J_EI, J_II=J_II, J_C=J_C,
                     W_E=W_E, W_I=W_I,
                     I_0A=I_0A, I_0B=I_0B, I_0C=I_0C,
                     GAMMA=GAMMA, GAMMA_I=GAMMA_I,
                     SIGMA_A=SIGMA_A, SIGMA_B=SIGMA_B,
                     N=N, Y=Y, S=S, I_long_A=I_long_A, I_long_B=I_long_B, I_long_C=I_long_C,
                     dt=dt, dt_over_taus=dt_over_taus, I_ext=I_ext,
                     I_mon=I_mon)
        # Update the state matrix using Euler–Maruyama integration
        Y += dY
        
        S += dS
        
        S_hist[k] = S  # new value at index k
        
        # Apply lesions (set the first six state variables to zero for specified indices)
        if lesion_indices.size > 0:
            if t >= lesion_start_time:
                for lesion_idx in lesion_indices:
                    Y[lesion_idx, :6] = 0

        # Save a copy of the current state in the history
        state_history.append(Y.copy())
        input_history.append(I_mon.copy())
        
    return state_history, Y, input_history, S_hist

# Function for BOLD signal, not used in this paper
@njit(nogil=True)
def simulation_run_SA_SB(Y, t_eval, dt, dt_over_taus, ts_ext_start, ts_ext_end, I_ext_strengths,
                    A, B, D, R_0, C_0, C_1, G_I, J_s, J_IE, J_C, J_EI, J_II, W_E, W_I, I_0A, I_0B, I_0C,
                    GAMMA, GAMMA_I, SIGMA_A, SIGMA_B, N, lesion_indices):
    # Euler-Maruyama integration
    Y = Y.copy()
    state_history = [Y.copy()[:, 3:5]]
    
    for i, t in enumerate(t_eval[1:], 1):
        # Generate I_ext
        I_ext = np.zeros((N, 3))
        if ts_ext_start is not None: # if external input is provided
            for j, (t_ext_start, t_ext_end) in enumerate(zip(ts_ext_start, ts_ext_end)): # loop over external input periods
                if t_ext_start <= t < t_ext_end:
                    I_ext += I_ext_strengths[j]

        # Calculate the differentials
        dY = calc_dY(A=A, B=B, D=D, R_0=R_0, C_0=C_0, C_1=C_1, G_I=G_I, J_s=J_s, J_IE=J_IE, J_EI=J_EI, J_II=J_II, J_C=J_C, W_E=W_E, W_I=W_I, I_0A=I_0A, I_0B=I_0B, I_0C=I_0C,
                     GAMMA=GAMMA, GAMMA_I=GAMMA_I, SIGMA_A=SIGMA_A, SIGMA_B=SIGMA_B, N=N, Y=Y, dt=dt, dt_over_taus=dt_over_taus, I_ext=I_ext)
        # Update the state
        Y += dY

        if lesion_indices.size > 0:
            for i in range(lesion_indices.size):
                idx = lesion_indices[i]
                Y[idx, :6] = 0
        if i % 2 == 0:
            # Store the S_A and S_B variables of current state
            state_history.append(Y.copy()[:, 3:5])

    return state_history, Y

class WorkingMemoryNetwork:
    def __init__(self, area_names, Y0, S0, delays, W, F, h,
                 normalized_nmda_rd, params,
                 gradient='NMDA', random_seed=None):
        """
        Initialize the network object for the working memory model.

        Parameters
        ----------
        area_names: list
            List of brain area names.
        Y0: np.ndarray
            N x 8 matrix of initial conditions. Each row is a brain area, each column is a variable vector (r_A, r_B, r_C, S_A, S_B, S_C, x_A, x_B).
        W: np.ndarray
            N x N matrix of connectivity strengths.
        F: np.ndarray
            N x N matrix of feedforward (FF) relative strengths (SLN matrix).
        h: np.ndarray
            1D array of the positions of the brain areas in the T1w:T2w ranking.
        params: object
            Parameters object containing the parameters of the model.
        random_seed: int, optional
            Random seed for reproducibility. Default is None.
        """
        self.area_names = area_names.copy()
        self.area_names = np.array(self.area_names)
        self.N = len(area_names) # number of brain areas

        # Check the dimensions of the input matrices
        assert Y0.shape == (self.N, 6), f"Y0 must be a {self.N} x 6 matrix!"
        assert W.shape == (self.N, self.N), f"W must be a {self.N} x {self.N} matrix!"
        assert F.shape == (self.N, self.N), f"F must be a {self.N} x {self.N} matrix!"
        assert self.N == h.shape[0], f"The length of h must be {self.N}!"
        assert np.min(W) >= 0, "Ensure all connections are non-negative!"
        assert np.all(np.diag(W) == 0), "Ensure no self-connections!"
        assert np.min(F) >= 0 and np.max(F) <= 1, "Ensure all feedforward relative strengths are between 0 and 1!"
        assert np.min(h) >= 0 and np.max(h) <= 1, "Ensure all hierarchical level values are between 0 and 1!"

        self.Y0 = Y0.copy() # store initial conditions
        self.S0 = S0.copy()
        self.S = S0.copy()
        self.delays = delays.copy()
        self.Y = Y0.copy() # initialize current state
        self.W = W.copy() # connectivity matrix
        self.W = np.array(W)
        self.F = F.copy() # FF relative strength matrix (SLN)
        self.F = np.array(F)

        if gradient == 'NMDA':
            self.h = normalized_nmda_rd.copy() # hierarchical level, if nmda_rd is used, define it with the normalized_rd number
        elif gradient == 'h':
            self.h = h.copy() # else, use the original hierarchy values as the gradient

        self.params = params
        self.J_s = calc_J_s(J_MIN=self.params.J_MIN, J_MAX=self.params.J_MAX, h=self.h) # calculate gradient of J_s        
        self.J_IE = calc_J_IE(J_0=params.J_0, J_C=params.J_C, J_EI=params.J_EI, ZETA=params.ZETA, J_s=self.J_s) # Calculate gradient of J_IE based on updated J_s
        self.lambda_I = 1 - self.F # Relative strength of (feedback) inhibitory projections
        # Precompute G * W * F and G/Z * W * (1-F) with gradients of long-range projection strengths
        self.W_E = self.params.G * self.W * np.outer(self.J_s / np.max(self.J_s), np.ones(self.N)) * self.F # Normalize by J_s / max(J_s)
        self.W_I = (self.params.ALPHA * self.params.G / self.params.Z) * self.W * np.outer(self.J_IE / np.max(self.J_IE), np.ones(self.N)) * self.lambda_I # Normalized by J_IE / max(J_IE)
        self.state_histories = [] # list to store the state history
        self.time_histories = [] # list to store the time history

        if random_seed:
            self.random_seed = random_seed
            self.rng = np.random.default_rng(random_seed)
        else:
            self.random_seed = None
            self.rng = np.random.default_rng()
        
        # Monitor for current input to the 3 populations of each node
        self.I_mon = np.zeros(shape=(self.N, 3))
        self.input_history = []

    def run(self, t_end, dt, t_start=0, I_ext_strengths=None, ts_ext_start=None, ts_ext_end=None,
            lesion_areas=None, lesion_start_time=0):
        """
        Run the simulation.

        Parameters
        ----------
        t_end: float
            End time of the simulation.
        dt: float
            Time step.
        t_start: float, optional
            Start time of the simulation. Default is 0.
        I_ext_strengths: np.ndarray, optional
            N_period x N x 3 matrix of external inputs strengths. N_period is the number of input periods. In each input period, the external input strength is a N x 3 matrix. If None, no external input.
        ts_ext_start: np.ndarray, optional
            Start times of the external input. If None, no external input.
        ts_ext_end: np.ndarray, optional
            End times of the external input. If None, no external input.
        lesion_areas: list, optional
            List of brain areas to lesion. If None, no lesion.

        Returns
        -------
        state_history: np.ndarray
            State history in this simulation (len(t_eval) x N x 8 array).
        """
        # Generate time points
        t_eval = np.arange(t_start, t_end+dt, dt)
        self.time_histories.append(t_eval)
        if lesion_areas is not None:
            # Precompute lesion indices only once (outside the @njit(nogil=True) function)
            lesion_indices = np.array([np.where(self.area_names == area)[0][0] for area in lesion_areas])
        else:
            lesion_indices = np.empty(0, dtype=np.int64)
        # Check if external input is provided
        if I_ext_strengths is None:
            ts_ext_start, ts_ext_end = None, None
        else:
            assert I_ext_strengths.shape[0] == len(ts_ext_start) == len(ts_ext_end), "Number of external input periods should be consistent!"
            assert I_ext_strengths.shape[1:] == (self.N, 3), "In each external input period, I_ext_strength should be an N x 3 matrix!"
            for t_ext_start, t_ext_end in zip(ts_ext_start, ts_ext_end):
                assert t_ext_start < t_ext_end, "ts_ext_start must be less than ts_ext_end!"
                assert t_ext_start >= t_start and t_ext_end <= t_end, "External input time must be within the simulation time!"

        # Precompute time step ratios
        dt_over_taus = precompute_dt_over_taus(TAU_R=self.params.TAU_R, TAU_N=self.params.TAU_N, TAU_G=self.params.TAU_G, TAU_NOISE=self.params.TAU_NOISE, dt=dt)
        # Convert arrays to contiguous objects so they can be efficiently accessed from memory
        self.W_E = np.ascontiguousarray(self.W_E)
        self.W_I = np.ascontiguousarray(self.W_I)
        state_history, self.Y, input_history, S_hist = simulation_run(Y=self.Y, S=self.S, delays=self.delays, t_eval=t_eval, dt=dt, dt_over_taus=dt_over_taus,
                                                ts_ext_start=ts_ext_start, ts_ext_end=ts_ext_end, I_ext_strengths=I_ext_strengths,
                                                A=self.params.A, B=self.params.B, D=self.params.D, R_0=self.params.R_0,
                                                C_0=self.params.C_0, C_1=self.params.C_1, G_I=self.params.G_I, J_s=self.J_s,
                                                J_IE=self.J_IE, J_C=self.params.J_C, J_EI=self.params.J_EI, J_II=self.params.J_II,
                                                W_E=self.W_E, W_I=self.W_I, I_0A=self.params.I_0A, I_0B=self.params.I_0B,
                                                I_0C=self.params.I_0C, GAMMA=self.params.GAMMA, GAMMA_I=self.params.GAMMA_I,
                                                SIGMA_A=self.params.SIGMA_A, SIGMA_B=self.params.SIGMA_B,
                                                N=self.N, lesion_indices=lesion_indices,
                                                lesion_start_time=lesion_start_time, I_mon=self.I_mon)
        
        # Store the state history
        self.state_histories.append(np.array(state_history))
        self.input_history.append(input_history)

        return np.array(state_history), np.array(input_history), S_hist
    
    # Function for BOLD signal, not used in this paper
    def run_SA_SB(self, t_end, dt, t_start=0, I_ext_strengths=None, ts_ext_start=None, ts_ext_end=None, lesion_areas=None):
        """
        Run the simulation.

        Parameters
        ----------
        t_end: float
            End time of the simulation.
        dt: float
            Time step.
        t_start: float, optional
            Start time of the simulation. Default is 0.
        I_ext_strengths: np.ndarray, optional
            N_period x N x 3 matrix of external inputs strengths. N_period is the number of input periods. In each input period, the external input strength is a N x 3 matrix. If None, no external input.
        ts_ext_start: np.ndarray, optional
            Start times of the external input. If None, no external input.
        ts_ext_end: np.ndarray, optional
            End times of the external input. If None, no external input.
        lesion_areas: list, optional
            List of brain areas to lesion. If None, no lesion.

        Returns
        -------
        state_history: np.ndarray
            State history in this simulation (len(t_eval) x N x 8 array).
        """
        # Generate time points
        t_eval = np.arange(t_start, t_end+dt, dt)
        self.time_histories.append(t_eval)
        if lesion_areas is not None:
            # Precompute lesion indices only once (outside the @njit(nogil=True) function)
            lesion_indices = np.array([np.where(self.area_names == area)[0][0] for area in lesion_areas])
        else:
            lesion_indices = np.empty(0, dtype=np.int64)
        # Check if external input is provided
        if I_ext_strengths is None:
            ts_ext_start, ts_ext_end = None, None
        else:
            assert I_ext_strengths.shape[0] == len(ts_ext_start) == len(ts_ext_end), "Number of external input periods should be consistent!"
            assert I_ext_strengths.shape[1:] == (self.N, 3), "In each external input period, I_ext_strength should be an N x 3 matrix!"
            for t_ext_start, t_ext_end in zip(ts_ext_start, ts_ext_end):
                assert t_ext_start < t_ext_end, "ts_ext_start must be less than ts_ext_end!"
                assert t_ext_start >= t_start and t_ext_end <= t_end, "External input time must be within the simulation time!"

        # Precompute time step ratios
        dt_over_taus = precompute_dt_over_taus(TAU_R=self.params.TAU_R, TAU_N=self.params.TAU_N, TAU_G=self.params.TAU_G, TAU_NOISE=self.params.TAU_NOISE, dt=dt)
        # Convert arrays to contiguous objects so they can be efficiently accessed from memory
        self.W_E = np.ascontiguousarray(self.W_E)
        self.W_I = np.ascontiguousarray(self.W_I)
        state_history, self.Y = simulation_run_SA_SB(Y=self.Y, t_eval=t_eval, dt=dt, dt_over_taus=dt_over_taus,
                                                     ts_ext_start=ts_ext_start, ts_ext_end=ts_ext_end, I_ext_strengths=I_ext_strengths,
                                                     A=self.params.A, B=self.params.B, D=self.params.D, R_0=self.params.R_0,
                                                     C_0=self.params.C_0, C_1=self.params.C_1, G_I=self.params.G_I, J_s=self.J_s,
                                                     J_IE=self.J_IE, J_C=self.params.J_C, J_EI=self.params.J_EI, J_II=self.params.J_II,
                                                     W_E=self.W_E, W_I=self.W_I, I_0A=self.params.I_0A, I_0B=self.params.I_0B,
                                                     I_0C=self.params.I_0C, GAMMA=self.params.GAMMA, GAMMA_I=self.params.GAMMA_I,
                                                     SIGMA_A=self.params.SIGMA_A, SIGMA_B=self.params.SIGMA_B,
                                                     N=self.N, lesion_indices=lesion_indices)
        
        # Store the state history
        self.state_histories.append(np.array(state_history))

        return np.array(state_history)

    def get_state_histories(self, variables=['all']):
        """
        Get the state histories of the network.

        Parameters
        ----------
        variables: list
            list of variables to return. If ['all'], return all variables.
        Returns
        -------
        state_histories: list
            List of state matrices for all simulations (length N_sim). Each element is a N_t x N x N_var matrix.
        """
        if 'all' in variables:
            return self.state_histories

        hash_map = {'r_A': 0, 'r_B': 1, 'r_C': 2, 'S_A': 3, 'S_B': 4, 'S_C': 5, 'x_A': 6, 'x_B': 7}
        indices = [hash_map[variable] for variable in variables]

        extracted_histories = []
        for state_history in self.state_histories:
            # state_matrix is of shape (time_steps, N, 8)
            # Extract the desired variables along the last axis
            extracted_history = state_history[:, :, indices]
            extracted_histories.append(extracted_history)

        return extracted_histories

    def clear_histories(self):
        """
        Clear the time histories and state histories.
        """
        self.state_histories = []
        self.time_histories = []

    def reset(self, Y0=None, random_seed=None):
        """
        Reset the network to a given initial state or the original initial state.

        Parameters
        ----------
        Y0: np.ndarray, optional
            N x 8 matrix of initial conditions. Each row is a brain area, each column is a variable vector (r_A, r_B, r_C, S_A, S_B, S_C, x_A, x_B). If None, use the original initial state.
        random_seed: int, optional
            Random seed for reproducibility. If None, use the original random seed.
        """
        self.Y0 = Y0 or self.Y0  # If Y0 is not provided, use the original initial state
        self.Y = self.Y0 # Reset the state
        self.clear_histories()
        if random_seed:
            self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed) # Reset the random number generator
    
    # Function for BOLD signal, not used in this paper
    def plot_all_areas_S_A_SB(self, xlim, ylim, axes_flat=None, variables=['S_A', 'S_B'], colors=[[.1, .6, .8], [.6, 0, .5]], title=None, save_path=None, downsample=10, legend=True, show=True):
            """
            Plot the variables of all brain areas.

            Parameters
            ----------
            xlim: tuple
                Tuple of x-axis limits (start, end).
            axes_flat: matplotlib.axes.Axes, optional
                Flattened axes object to plot on. If None, create a new figure.
            variables: list of str
                List of variables to plot. Must match variable names in state history.
            colors: list
                List of colors for each variable, must match the length of `variables`.
            title: str, optional
                Title of the plot.
            save_path: str, optional
                Path to save the plot. If None, do not save the plot.
            downsample: int, optional
                Downsampling factor for time series data to speed up plotting. Default is 10.
            legend: bool, optional
                Whether to show the legend. Default is True.

            Returns
            -------
            lines: list
                List of Line2D objects representing the plotted data.
            """
            # Check variables and colors
            if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
                raise ValueError("`variables` must be a list of strings representing variable names.")
            assert len(variables) == len(colors), "Length of variables and colors must be the same."

            # Prepare data and area names
            state_history = self.get_state_histories_SA_SB(variables)[-1]
            t = self.time_histories[-1] # Time array
            t = t[::2]
            t_downsampled = t[::downsample] # Downsample time
            t_end = t[-1] # Use the last time point as t_end

            # Prepare plotting
            rows = math.ceil(self.N / 6) # Determine grid rows
            existing_plots = False # Track if there are existing plots in the axes
            if axes_flat is None:
                fig, axes = plt.subplots(rows, 6, figsize=(15, 2 * rows), dpi=100)
                fig.subplots_adjust(hspace=0.5, wspace=0.3)
                axes = axes.flatten() # Flatten for easy iteration
            else:
                axes = axes_flat
                if axes[0].lines:
                    existing_plots = True

            # Plot firing rates for each brain area
            lines = []
            for i, ax in enumerate(axes[:self.N]): # Only iterate over valid brain areas
                ymax = 0 # Track max y value for this subplot
                for var_idx, var in enumerate(variables):
                    data = state_history[:, i, var_idx][::downsample] # Downsample data
                    if i == 0:
                        line = ax.plot(t_downsampled, data, label='$'+var+'$', color=colors[var_idx], linewidth=2)
                        lines.append(line[0])
                        if legend:
                            ax.legend(loc="upper right", fontsize=10)
                    else:
                        ax.plot(t_downsampled, data, color=colors[var_idx], linewidth=2)
                    ymax = max(ymax, data.max()) # Update y max value

                # Add titles, labels, and ylimits
                ax.set_title(self.area_names[i], fontsize=13, fontweight='bold')
                if existing_plots: # If there are existing plots before calling this function
                    x_min1, x_max1 = ax.get_xlim()
                    x_min2, x_max2 = xlim
                    ax.set_xlim(min(x_min1, x_min2), max(x_max1, x_max2))
                    y_min1, y_max1 = ax.get_ylim()
                    y_min2, y_max2 = -max(ymax * 0.2, 0.5), max(ymax * 1.3, 2)
                    ax.set_ylim(min(y_min1, y_min2), max(y_max1, y_max2))
                else: # If there are no existing plots before calling this function
                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)
                    #ax.set_ylim(-max(ymax * 0.2, 0.5), max(ymax * 1.3, 2))

                if i % 6 == 0:  # First column: add ylabel
                    ax.set_ylabel("Rate (sp/s)", fontsize=12)
                if i >= (rows - 1) * 6:  # Last row: add xlabel
                    ax.set_xlabel("Time (s)", fontsize=12)

            # Remove empty subplots if self.N is not a multiple of 6
            for j in range(self.N, len(axes)):
                fig.delaxes(axes[j])

            # Adjust layout and add optional title
            if title:
                plt.suptitle(title, fontsize=18, fontweight='bold')
            plt.tight_layout()

            # Save the figure if a path is provided
            if save_path:
                plt.savefig(save_path, dpi=300)  # Set proper DPI for high-quality saving

            if show == True:
                plt.show()
            plt.close()
            return lines
        
    def plot_all_areas(self, xlim, variables=['r_A', 'r_B'], colors=[[.1, .6, .8], [.6, 0, .5]], title=None, save_path=None, downsample=10, legend=True, show=True):
        """
        Plot the variables of all brain areas.

        Parameters
        ----------
        xlim: tuple
            Tuple of x-axis limits (start, end).
        variables: list of str
            List of variables to plot. Must match variable names in state history.
        colors: list
            List of colors for each variable, must match the length of `variables`.
        title: str, optional
            Title of the plot.
        save_path: str, optional
            Path to save the plot. If None, do not save the plot.
        downsample: int, optional
            Downsampling factor for time series data to speed up plotting. Default is 10.
        legend: bool, optional
            Whether to show the legend. Default is True.
        show: bool, optional
            Whether to show the figure. Default is True.
        Returns
        -------
        lines: list
            List of Line2D objects representing the plotted data.
        """
        # Check variables and colors
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            raise ValueError("`variables` must be a list of strings representing variable names.")
        assert len(variables) == len(colors), "Length of variables and colors must be the same."

        # Prepare data and area names
        state_history = self.get_state_histories(variables)[-1]
        t = self.time_histories[-1] # Time array
        t_downsampled = t[::downsample] # Downsample time

        # Prepare plotting
        rows = math.ceil(self.N / 6) # Determine grid rows
        fig, axes = plt.subplots(rows, 6, figsize=(15, 2 * rows), dpi=100)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        axes = axes.flatten() # Flatten for easy iteration

        # Plot firing rates for each brain area
        lines = []
        for i, ax in enumerate(axes[:self.N]): # Only iterate over valid brain areas
            ymax = 0 # Track max y value for this subplot
            for var_idx, var in enumerate(variables):
                data = state_history[:, i, var_idx][::downsample] # Downsample data
                if i == 0:
                    line = ax.plot(t_downsampled - 2, data, label='$'+var+'$', color=colors[var_idx], linewidth=2)
                    lines.append(line[0])
                    if legend:
                        ax.legend(loc="upper right", fontsize=10)
                else:
                    ax.plot(t_downsampled - 2, data, color=colors[var_idx], linewidth=2)
                ymax = max(ymax, data.max()) # Update y max value

            # Add titles, labels, and ylimits
            ax.set_title(self.area_names[i], fontsize=13, fontweight='bold')
            ax.set_xlim(*xlim)
            # ax.set_ylim(*ylim)
            ax.set_ylim(-max(ymax * 0.2, 0.5), max(ymax * 1.3, 2))

            if i % 6 == 0: # First column: add ylabel
                ax.set_ylabel("Firing Rate (Hz)", fontsize=12)
            if i >= (rows - 1) * 6: # Last row: add xlabel
                ax.set_xlabel("Time (s)", fontsize=12)
            
            ax = plt.gca()
            for spine in ['top','right']:
                ax.spines[spine].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.grid(linestyle=':', alpha=0.5)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')

        # Remove empty subplots if self.N is not a multiple of 6
        for j in range(self.N, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and add optional title
        if title:
            plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)

        if show == True:
            plt.show()
        plt.close()
        return lines
    
    def plot_all_areas_current(self, xlim, ylim, variables=['I_A', 'I_B', 'I_C'], colors=[[.1, .6, .8], [.6, 0, .5], [.3, .2, .5]], title=None, save_path=None, downsample=10, legend=True, show=True):
        """
        Plot the variables of all brain areas.

        Parameters
        ----------
        xlim: tuple
            Tuple of x-axis limits (start, end).
        variables: list of str
            List of variables to plot. Must match variable names in state history.
        colors: list
            List of colors for each variable, must match the length of `variables`.
        title: str, optional
            Title of the plot.
        save_path: str, optional
            Path to save the plot. If None, do not save the plot.
        downsample: int, optional
            Downsampling factor for time series data to speed up plotting. Default is 10.
        legend: bool, optional
            Whether to show the legend. Default is True.
        show: bool, optional
            Whether to show the figure. Default is True.
        Returns
        -------
        lines: list
            List of Line2D objects representing the plotted data.
        """
        # Check variables and colors
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            raise ValueError("`variables` must be a list of strings representing variable names.")
        assert len(variables) == len(colors), "Length of variables and colors must be the same."

        # Prepare data and area names
        state_history = np.array(self.input_history).squeeze(axis=0)
        print("history.shape:", state_history.shape)
        t = self.time_histories[-1] # Time array
        t_downsampled = t[::downsample] # Downsample time

        # Prepare plotting
        rows = math.ceil(self.N / 6) # Determine grid rows
        fig, axes = plt.subplots(rows, 6, figsize=(15, 2 * rows), dpi=100)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        axes = axes.flatten() # Flatten for easy iteration

        # Plot firing rates for each brain area
        lines = []
        for i, ax in enumerate(axes[:self.N]): # Only iterate over valid brain areas
            ymax = 0 # Track max y value for this subplot
            for var_idx, var in enumerate(variables):
                data = state_history[:, i, var_idx][::downsample] # Downsample data
                if i == 0:
                    line = ax.plot(t_downsampled, data, label='$'+var+'$', color=colors[var_idx], linewidth=2)
                    lines.append(line[0])
                    if legend:
                        ax.legend(loc="upper right", fontsize=10)
                else:
                    ax.plot(t_downsampled, data, color=colors[var_idx], linewidth=2)
                ymax = max(ymax, data.max()) # Update y max value

            # Add titles, labels, and ylimits
            ax.set_title(self.area_names[i], fontsize=13, fontweight='bold')
            ax.set_xlim(*xlim)
            if ylim:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(-max(ymax * 0.2, 0.5), max(ymax * 1.3, 2))

            if i % 6 == 0: # First column: add ylabel
                ax.set_ylabel("Rate (Hz)", fontsize=12)
            if i >= (rows - 1) * 6: # Last row: add xlabel
                ax.set_xlabel("Time (s)", fontsize=12)

        # Remove empty subplots if self.N is not a multiple of 6
        for j in range(self.N, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and add optional title
        if title:
            plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)

        if show == True:
            plt.show()
        plt.close()
        return lines

    def plot_n_areas(self,
                xlim,
                ylim=None,
                target_area_names=['Cont_Cing_1', 'SomMot_10', 'SomMot_7'],
                names_for_figures=None,
                variables=['r_A', 'r_B'],
                colors=[[.1, .6, .8], [.6, 0, .5]],
                title=None,
                save_path=None,
                downsample=10,
                legend=True,
                show=True):
        """
        Plot and save the firing rates of selected brain areas as separate figures,
        with the first two areas using the provided ylim, and subsequent areas capped at 20 Hz.

        Parameters
        ----------
        xlim: tuple
            Tuple of x-axis limits (start, end).
        ylim: tuple, optional
            Tuple of y-axis limits for the first two areas (start, end). Subsequent areas will use (0,20).
        target_area_names: list, optional
            List of brain area names to plot.
        variables: list of str
            List of variables to plot. Must match variable names in state history.
        colors: list
            List of colors for each variable, must match the length of `variables`.
        title: str, optional
            Title of the plot (used as prefix for each figure).
        save_path: str, optional
            Base path or filename. Each area will be saved as '<base>_<area>.png'.
        downsample: int, optional
            Downsampling factor for time series data to speed up plotting. Default is 10.
        legend: bool, optional
            Whether to show the legend. Default is True.
        show: bool, optional
            Whether to display each figure interactively. Default is True.

        Returns
        -------
        lines: list
            List of Line2D objects for the first area.
        """
        import matplotlib as mpl
        mpl.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        })
        # Validate inputs
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            raise ValueError("`variables` must be a list of strings representing variable names.")
        assert len(variables) == len(colors), "Length of variables and colors must be the same."

        # Prepare data
        state_history = self.get_state_histories(variables)[-1]
        t = self.time_histories[-1]
        t_down = t[::downsample]

        lines = []

        # Loop over each area and create its own figure
        for i, area_name in enumerate(target_area_names):
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(1.65, 1.55))
            ymax = 0

            # Find area index
            area_idx = np.where(np.array(self.area_names) == area_name)[0][0]

            # Plot each variable
            for var_idx, var in enumerate(variables):
                data = state_history[:, area_idx, var_idx][::downsample]
                line, = ax.plot(t_down-2, data, label=f'$ {var} $', color=colors[var_idx], linewidth=2)
                if i == 0:
                    lines.append(line)
                ymax = max(ymax, data.max())

            # Legend
            if i == 0:
                if legend:
                    ax.legend(loc='upper right')

            # Titles and labels
            ax.set_title(names_for_figures[i])
            ax.set_xlim(*xlim)
            
            if ylim:
                if i < 2:
                    ax.set_ylim(*ylim)
                else:
                    ax.set_ylim(-5, 30)
            else:
                ax.set_ylim(-max(ymax * 0.2, 0.5), max(ymax * 1.3, 2))

            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_xlabel('Time (s)')
            ax = plt.gca()
            from matplotlib.ticker import MaxNLocator
            ax.minorticks_off()
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            for spine in ['top','right']:
                ax.spines[spine].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.grid(linestyle=':', alpha=0.5)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')

            if title:
                fig.suptitle(title, fontweight='bold')

            fig.tight_layout()

            # Save if requested
            if save_path:
                base, ext = os.path.splitext(save_path)
                out_path = f"{base}_{area_name}{ext or '.png'}"
                fig.savefig(out_path)

            # Show and close
            if show:
                plt.show()
            plt.close(fig)

        return lines



    def check_sustained_activity_after_stim_removal(self):
        """
        Check if any brain areas exhibit persistent activity by checking if firing rates 
        are significantly higher in the last 2 seconds of the simulation.

        Returns
        -------
        (bool, list of tuples)
            - bool: True if persistent activity is detected for at least one population, 
                    False otherwise.
            - list of tuples: Each tuple contains:
                - Population name ('r_A' or 'r_B')
                - List of brain area indices that exhibit persistent activity
        """
        # Extract time and state histories
        t = self.time_histories[-1] # Time array
        state_history = self.get_state_histories(['r_A', 'r_B'])[-1] # Firing rate data

        # Get indices for the first 2 seconds and the last 2 seconds
        first_2s_indices = np.where(t < 2)[0]
        last_2s_indices = np.where(t > t[-1] - 2)[0]

        if len(first_2s_indices) == 0 or len(last_2s_indices) == 0:
            print("Not enough data points in the first or last 2 seconds to perform the analysis.")
            return False, []

        # Compute average firing rates for the first and last 2 seconds
        first_2s_avg = np.mean(state_history[first_2s_indices, :, :], axis=0)
        last_2s_avg = np.mean(state_history[last_2s_indices, :, :], axis=0)

        # Keep track of populations that show persistent activity
        changed_populations = []

        # Compare firing rates to detect persistent activity
        for pop_idx, pop_name in enumerate(['r_A', 'r_B']):
            # Condition for persistent activity: Either a significant increase (>5 Hz) or sustained high activity (>10 Hz) which could have occurred from the start
            persistent_indices = np.where(
                ((last_2s_avg[:, pop_idx] - first_2s_avg[:, pop_idx]) > 5) # Increased activity
                | (last_2s_avg[:, pop_idx] > 10) # Sustained high activity
            )[0]

            if persistent_indices.size > 0:
                # print(
                #     f"Population {pop_name}: Brain nodes with indices {persistent_indices.tolist()} "
                #     f"exhibit persistent activity. Areas: {self.area_names[persistent_indices]}"
                # )
                changed_populations.append((pop_name, persistent_indices.tolist())) # Store as tuple
            # else:
            #     # print(f"Population {pop_name}: No brain nodes exhibit persistent activity.")

        changes_detected = (len(changed_populations) > 0)
        return changes_detected, changed_populations
    
    def check_spontaneous_switch_to_high_state(self, t_stim_start):
        """
        Check if any brain areas spontaneously move to the higher attractor at any point before
        the stimulus is given (0.5s after simulation start and 0.1s before stimulus).

        Returns
        -------
        (bool, list of tuples)
            - bool: True if high activity is detected for at least one population, 
                    False otherwise.
            - list of tuples: Each tuple contains:
                - Population name ('r_A' or 'r_B')
                - List of brain area indices that exhibit persistent activity
        """
        # Extract time and state histories
        t = self.time_histories[-1] # Time array
        state_history = self.get_state_histories(['r_A', 'r_B'])[-1] # Firing rate data

        # Get indices for the between 0.5s and 0.1s before stimulus onset
        before_stim_indices = np.where((t > 0.5) & (t < (t_stim_start - 0.1)))[0]

        # Compute maximum firing rates after 0.5s (steady state) and before the stimulus onset
        before_stim_max = np.max(state_history[before_stim_indices, :, :], axis=0)

        # Keep track of populations that show persistent activity
        changed_populations = []

        # Compare firing rates to detect persistent activity
        for pop_idx, pop_name in enumerate(['r_A', 'r_B']):
            # Condition for persistent activity: high activity (>8 Hz) which could have occurred from the start
            persistent_indices = np.where(
                (before_stim_max[:, pop_idx] > 8) # Sustained high activity
            )[0]

            if persistent_indices.size > 0:
                # print(
                #     f"Population {pop_name}: Brain nodes with indices {persistent_indices.tolist()} "
                #     f"exhibit persistent activity. Areas: {self.area_names[persistent_indices]}"
                # )
                changed_populations.append((pop_name, persistent_indices.tolist())) # Store as tuple
            # else:
            #     # print(f"Population {pop_name}: No brain nodes exhibit persistent activity.")

        changes_detected = (len(changed_populations) > 0)
        return changes_detected, changed_populations
    
    def plot_avg_firing_rates_last_two_seconds(self, variables=['r_A', 'r_B'], color=['green', 'red'], save_path=None, show=True):
        """
        Plot the average firing rates over the last two seconds of the simulation for all brain areas.

        Parameters
        ----------
        variables: list of str
            List of variables to plot. Must match variable names in state history.
        color: str
            Color of the plot.
        save_path: str, optional
            Path to save the plot. If None, do not save the plot.

        Returns
        -------
        None
        """
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        })
        # Check variables
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            raise ValueError("`variables` must be a list of strings representing variable names.")
        
        # Extract time and state histories
        t = self.time_histories[-1] # Time array
        state_history = self.get_state_histories(variables)[-1] # Firing rate data

        # Identify the indices for the last two seconds
        end_time = t[-1]
        start_time = end_time - 2 # Last two seconds
        indices = np.where((t >= start_time) & (t <= end_time))[0]

        if len(indices) == 0:
            raise ValueError("No data points found in the last two seconds of the simulation.")

        # Compute the average firing rates for variable r_A over the last two seconds
        avg_firing_rates_rA = np.mean(state_history[indices, :, 0], axis=0)
        avg_firing_rates_rA.sort()
        # Compute the average firing rates for variable r_B over the last two seconds
        avg_firing_rates_rB = np.mean(state_history[indices, :, 1], axis=0)
        avg_firing_rates_rB.sort()
        
        # Plot the data
        fig, ax = plt.subplots(figsize=(2.2, 1.9)) # Increase dpi for higher quality
        ax.plot(range(len(avg_firing_rates_rA)), avg_firing_rates_rA, 'o', color=color[0], markersize=1.5, label='r_A')
        ax.plot(range(len(avg_firing_rates_rB)), avg_firing_rates_rB, 'o', color=color[1], markersize=1.5, label='r_B')
        ax.set_xlabel("Area")
        ax.set_ylabel("Firing Rate (Hz)")
        # ax.set_title(f"Average firing rates in the last 2s", fontsize=16, fontweight='bold')
        plt.tight_layout()
        ax = plt.gca()
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.grid(linestyle=':', alpha=0.5)
        ax.tick_params(axis='both', which='major')
        ax.tick_params(axis='both', which='minor')
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=1200)
        if show == True:
            plt.show()
        plt.close()
    
    def plot_avg_firing_rates_last_two_seconds_if_discontinuity(self, variables=['r_A', 'r_B'], color=['green', 'red'], save_path=None, show=True):
        """
        Plot the average firing rates over the last two seconds of the simulation for all brain areas.

        Parameters
        ----------
        variables: list of str
            List of variables to plot. Must match variable names in state history.
        color: str
            Color of the plot.
        save_path: str, optional
            Path to save the plot. If None, do not save the plot.

        Returns
        -------
        None
        """
        # Check variables
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            raise ValueError("`variables` must be a list of strings representing variable names.")
        
        # Extract time and state histories
        t = self.time_histories[-1] # Time array
        state_history = self.get_state_histories(variables)[-1] # Firing rate data

        # Identify the indices for the last two seconds
        end_time = t[-1]
        start_time = end_time - 2 # Last two seconds
        indices = np.where((t >= start_time) & (t <= end_time))[0]

        if len(indices) == 0:
            raise ValueError("No data points found in the last two seconds of the simulation.")

        # Compute the average firing rates for variable r_A over the last two seconds
        avg_firing_rates_rA = np.mean(state_history[indices, :, 0], axis=0)
        avg_firing_rates_rA.sort()
        # Compute the average firing rates for variable r_B over the last two seconds
        avg_firing_rates_rB = np.mean(state_history[indices, :, 1], axis=0)
        avg_firing_rates_rB.sort()

        for index, avg_A_sorted in enumerate(avg_firing_rates_rA):
            if index == 0:
                pass
            elif avg_firing_rates_rA[index] - avg_firing_rates_rA[index - 1] > 3:
                # Plot the data
                plt.figure(figsize=(8, 6), dpi=100) # Increase dpi for higher quality
                plt.plot(range(len(avg_firing_rates_rA)), avg_firing_rates_rA, 'o', color=color[0], markersize=5, label='r_A')
                plt.plot(range(len(avg_firing_rates_rB)), avg_firing_rates_rB, 'o', color=color[1], markersize=5, label='r_B')
                plt.xlabel("Area", fontsize=20)
                plt.ylabel("Firing Rate (Hz)", fontsize=20)
                plt.xticks(fontsize=16)  # Set font size for x-axis ticks
                plt.yticks(fontsize=16)  # Set font size for y-axis ticks
                plt.title(f"Average firing rates in the last 2s", fontsize=16, fontweight='bold')
                plt.legend()
                plt.tight_layout()
                
                # Save the plot if a path is provided
                if save_path:
                    plt.savefig(save_path, dpi=100)
                if show == True:
                    plt.show()
                plt.close()
    
    def save_avg_firing_rates_last_two_seconds(self, variables=['r_A']):
        """
        Return the average firing rates (of pop A) over the last two seconds of the simulation for all brain areas.

        Parameters
        ----------
        variables: list of str
            List of variables to plot. Must match variable names in state history.

        Returns
        -------
        None
        """
        # Check variables
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            raise ValueError("`variables` must be a list of strings representing variable names.")
        
        # Extract time and state histories
        t = self.time_histories[-1] # Time array
        state_history = self.get_state_histories(variables)[-1] # Firing rate data

        # Identify the indices for the last two seconds
        end_time = t[-1]
        start_time = end_time - 2 # Last two seconds
        indices = np.where((t >= start_time) & (t <= end_time))[0]

        if len(indices) == 0:
            raise ValueError("No data points found in the last two seconds of the simulation.")

        if variables == ['r_A']:
            # Compute the average firing rates over the last two seconds of population A
            avg_firing_rates = np.mean(state_history[indices, :, 0], axis=0)
        
            return avg_firing_rates
        
        elif variables == ['r_A', 'r_B']:
            # Compute the average firing rates over the last two seconds of population A and B
            avg_firing_rates_A, avg_firing_rates_B = np.mean(state_history[indices, :, 0], axis=0), np.mean(state_history[indices, :, 1], axis=0)

            return avg_firing_rates_A, avg_firing_rates_B