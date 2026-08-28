import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from tqdm import tqdm
from .WM_network import WorkingMemoryNetwork


def dxdt(t, x, tau_noise, sigma):
    """
    Compute the derivative of the OU process.

    Parameters
    ----------
    t : float
        Time
    x : np.ndarray
        1D array of length N
    tau_noise : float
        Time constant of the noise
    sigma : float
        Standard deviation of the noise term

    Returns
    -------
    dxdt : np.ndarray
        1D array of length N
    """
    return -x / tau_noise + sigma * np.random.normal(0, 1, len(x)) / np.sqrt(tau_noise)


def generate_noise_function(N, t_eval, tau_noise, sigma):
    """
    Generate a noise function for a fixed sigma and return an interpolated function.

    Parameters
    ----------
    N : int
        Number of brain areas
    t_eval : np.ndarray
        Time points at which to save the noise (1D array)
    tau_noise : float
        Time constant of the noise
    sigma : float
        Standard deviation of the noise term

    Returns
    -------
    interp_func : callable
        A function that takes time t and returns noise values
    """
    # Generate OU noise using solve_ivp
    sol = solve_ivp(
        dxdt,
        [t_eval[0], t_eval[-1]],
        y0=np.zeros(N),
        t_eval=t_eval,
        args=(tau_noise, sigma),
    )
    noise = sol.y.T  # Shape: (N_t, N)

    # Create an interpolation function for the noise
    interp_func = interp1d(
        t_eval, noise, axis=0, kind="linear", fill_value="extrapolate"
    )

    return interp_func


def correct_spine_counts(spine_counts):
    """
    Correct the spine counts based on age factors. For ages 5 and 10, the spine counts are increased by 15% and 30%, respectively.

    Parameters
    ----------
    spine_counts : np.ndarray
        Spine counts

    Returns
    -------
    corrected_spine_counts : np.ndarray
    """
    # age factor
    AF = np.ones(30)
    AF[[13, 14, 15, 16, 18, 29]] = np.multiply(AF[[13, 14, 15, 16, 18, 29]], 1.15)  # 5yo, 15% increase
    AF[[5, 7, 20]] = np.multiply(AF[[5, 7, 20]], 1.30)  # 10yo, 30% increase

    return spine_counts * AF


def load_and_preprocess_data(data_folder_path=None):
    """
    Load and preprocess the data.

    Parameters
    ----------
    data_folder_path : str, optional
        Path to the data folder containing the data files. If None, the relative path is used.

    Returns
    -------
    area_names : np.ndarray
        Array of area names
    area_lobes : np.ndarray
        Array of area lobes
    area_descriptions : np.ndarray
        Array of area descriptions
    SLN : np.ndarray
        Structural connectivity matrix
    W : np.ndarray
        Weight matrix. W_ij = 1.2 * FLN_ij^0.3 / sum_j(1.2 * FLN_ij^0.3)
    hier_vals : np.ndarray
        Hierarchy values. If spine counts are negative (missing), the original hierarchy values are used.
    """
    if data_folder_path is None:
        data_folder_path = "data"

    areas_file = data_folder_path + '/raw/macaque/areaList.xlsx'
    df = pd.read_excel(areas_file, header = 0)
    area_names = df['Name'].to_numpy()
    area_lobes = df['Lobe'].to_numpy()
    area_descriptions = df['Description'].to_numpy()

    sln_file = data_folder_path + '/processed/macaque/macaque_t1_t2_SLN.xlsx'
    df = pd.read_excel(sln_file, header=None)
    SLN = df.to_numpy()

    fln_file = data_folder_path + '/raw/macaque/fln.xlsx'
    df = pd.read_excel(fln_file, header=None)
    FLN = df.to_numpy()
    W = 1.2 * FLN ** 0.3
    W /= np.sum(W, axis=1, keepdims=True)

    hier_file = data_folder_path + '/processed/macaque/macaque_hierarchy_from_t1_t2.xlsx'
    df = pd.read_excel(hier_file, header=None)
    hier_vals1 = df.to_numpy().flatten()    # hier_vals1 is the original hierarchy values

    spinec_file = data_folder_path + '/raw/macaque/spineCounts_ageCorrected.xlsx'
    df = pd.read_excel(spinec_file, header=None)
    spinec = df.to_numpy().flatten()

    hier_vals = spinec / np.max(spinec)
    indices = np.where(spinec < 0)
    hier_vals[indices] = hier_vals1[indices]    # replace empty spine counts with original hierarchy values

    return {'area_names': area_names, 'area_lobes': area_lobes, 'area_descriptions': area_descriptions, 'SLN': SLN, 'W': W, 'hier_vals': hier_vals}


def get_success_rate(desc, WM_network, t_end, dt, I_ext_strengths, ts_ext_start, ts_ext_end, runs=50, threshold=10, monitor_areas_idx=range(12, 30)):
    """
    Get the success rates of the WM model for a given year.

    Parameters
    ----------
    desc : str
        Description of the simulation
    WM_network : WorkingMemoryNetwork
        Working Memory network
    t_end : float
        End time of the simulation
    dt : float
        Time step
    I_ext_strengths : np.ndarray
        External input strengths
    ts_ext_start : np.ndarray
        Start times of the external inputs
    ts_ext_end : np.ndarray
        End times of the external inputs
    runs : int, optional
        Number of runs
    threshold : float, optional
        Threshold for distinguishing successful trials
    monitor_areas_idx : np.ndarray, optional
        Indices of areas to monitor and check for success

    Returns
    -------
    success_rate : float
        Success rate
    """
    success_times = 0
    for seed in tqdm(range(runs), desc=desc):
        WM_network.reset(random_seed=seed)
        state_history = WM_network.run(t_end=t_end, dt=dt, I_ext_strengths=I_ext_strengths, ts_ext_start=ts_ext_start, ts_ext_end=ts_ext_end)
        final_r_A = np.mean(state_history[int(-1/dt), monitor_areas_idx, 0], axis=0)    # mean r_A in the last one second for each area
        if np.sum(final_r_A > threshold) > len(monitor_areas_idx) / 2:
            success_times += 1
    return success_times / runs