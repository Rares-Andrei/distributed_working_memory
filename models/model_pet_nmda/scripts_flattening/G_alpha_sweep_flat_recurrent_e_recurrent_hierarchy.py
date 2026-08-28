"""
Script to:
1) Run a parameter sweep over (G, alpha) during working memory task.
2) Collect detailed statistics about persistent activity (probabilities, mean active areas, conditional means, etc.).
3) Track area-by-area frequency of persistent activity for each population.
4) Save results to CSV and generate plots.

Usage:
    python G_alpha_sweep_flat_recurrent_e_recurrent_hierarchy.py <G> <alpha>
"""
import os
import hashlib
import sys
import json
import numpy as np
import pandas as pd

# Add the project root to the PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.data import load_and_preprocess_data as get_data
from src.models import WorkingMemoryNetwork
from config.WM_params import WorkingMemoryParameters

if len(sys.argv) != 3:
    print("Usage: G_alpha_sweep_flat_recurrent_e_recurrent_hierarchy.py <G> <alpha>")
    sys.exit(1)

# Parse command-line arguments
G_var = float(sys.argv[1])
alpha = float(sys.argv[2])

def seed_for(G, alpha, run_idx, base=123456):
    key = f"{base}|G={G:.9f}|alpha={alpha:.9f}|run={run_idx}"
    h = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") # 64-bit int

# Prepare output directories
run_folder_name = f"G{G_var:.3f}_alpha{alpha:.3f}"
results_dir = os.path.join("../", "results", "g_alpha_flat_recurrent_e_recurrent_hierarchy", run_folder_name)
data_dir = os.path.join(results_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# Simulation timing
t_end = 15.0  # Total simulation time (s)
dt = 0.5e-3  # Time step (s)
t_eval = np.arange(0, t_end + dt, dt)
stim_start_time = 3.0
stim_duration = 0.5

# Seeds and network parameters
num_runs = 10
grad = 'h'
j_max = 0.3  # Fixed J_max over simulations
target_names = ['Vis_5', 'Vis_7', 'Vis_9']

# Log file for simulation details
sim_details_log_path = os.path.join(results_dir, "simulation_details.log")
with open(sim_details_log_path, "w") as L:
    L.write(f"Parameters: G={G_var}, alpha={alpha}, jmax={j_max}\n")
    L.write(f"# seeds: {num_runs}\n\n")

# Load data for simulation
data = get_data(data_folder_path='../../../data/processed', parcellation='schaefer200')
area_names = data['area_names']
SLN = data['SLN']
W = data['W']
h = data['hier_vals']
normalized_nmda_rd = data['nmda_rd']
N = len(area_names)

# External input setup
I_ext_strength = np.zeros((N, 3))
rows = np.where(np.isin(area_names, target_names))[0]
I_ext_strength[rows, 0] = 0.3
ts_ext_start = np.array([stim_start_time])
ts_ext_end = np.array([stim_start_time + stim_duration])

# Initialize accumulators
count_no_persist = count_A_only = count_B_only = count_A_and_B = 0
total_areas_A = total_areas_B = 0
total_areas_A_cond = total_areas_B_cond = 0
count_runs_A_active = count_runs_B_active = 0
count_spont_switch = 0
spont_switch_details_list = []
area_count_A = np.zeros(N, dtype=int)
area_count_B = np.zeros(N, dtype=int)
aggregated_results = []
area_freq_results = []
avg_firing_rates = []

# Main loop over seeds
for seed_idx in range(num_runs):
    seed = seed_for(G_var, alpha, seed_idx)
    rng_local = np.random.default_rng(seed)

    # Random initial conditions
    Y0_run = np.zeros((N, 8))
    Y0_run[:, 0:3] = 5 * (1 + np.tanh(rng_local.normal(0, 2, (N, 3))))

    # Update parameters
    WM_params = WorkingMemoryParameters()
    WM_params.update(
        G=G_var,
        ALPHA=alpha,
        J_MIN=0.4,
        J_MAX=0.4
    )

    # Log network creation
    with open(sim_details_log_path, "a") as sim_log:
        sim_log.write(f"Seed {seed_idx}: G={G_var:.3f}, alpha={alpha:.3f}, J_MAX={j_max:.3f}\n")

    # Instantiate and run network
    WM_network = WorkingMemoryNetwork(
        area_names=area_names,
        Y0=Y0_run,
        W=W,
        F=SLN,
        h=h,
        normalized_nmda_rd=normalized_nmda_rd,
        params=WM_params,
        gradient=grad,
        random_seed=seed
    )
    WM_network.run(
        t_end=t_end,
        dt=dt,
        I_ext_strengths=np.array([I_ext_strength]),
        ts_ext_start=ts_ext_start,
        ts_ext_end=ts_ext_end
    )

    # Analyze persistent activity
    changes_detected, changed_pops = WM_network.check_sustained_activity_after_stim_removal()
    changed_names = [p[0] for p in changed_pops]
    changed_areas = {p[0]: p[1] for p in changed_pops}
    num_A = len(changed_areas.get("r_A", []))
    num_B = len(changed_areas.get("r_B", []))

    # Unconditional & conditional area counts
    total_areas_A += num_A
    total_areas_B += num_B
    if "r_A" in changed_names:
        count_runs_A_active += 1
        total_areas_A_cond += num_A
    if "r_B" in changed_names:
        count_runs_B_active += 1
        total_areas_B_cond += num_B

    # Area-frequency tallies
    for idx in changed_areas.get("r_A", []):
        area_count_A[idx] += 1
    for idx in changed_areas.get("r_B", []):
        area_count_B[idx] += 1

    # Outcome counters
    if not changes_detected:
        count_no_persist += 1
    else:
        if "r_A" in changed_names and "r_B" in changed_names:
            count_A_and_B += 1
        elif "r_A" in changed_names:
            count_A_only += 1
        elif "r_B" in changed_names:
            count_B_only += 1
        else:
            count_no_persist += 1

    # Spontaneous switches
    spont_detected, spont_details = WM_network.check_spontaneous_switch_to_high_state(stim_start_time)
    if spont_detected:
        count_spont_switch += 1
    spont_switch_details_list.append(spont_details)

    # Save average firing rates
    avg_fr = WM_network.save_avg_firing_rates_last_two_seconds()
    avg_firing_rates.append(avg_fr)

    WM_network.reset()

# Aggregate over seeds
avg_firing_rate = np.mean(np.stack(avg_firing_rates, axis=0), axis=0)
np.save(os.path.join(data_dir, f'delay_FR_G{G_var:.3f}_alpha{alpha:.3f}_jmax{j_max:.3f}.npy'),
        avg_firing_rate)

p_no = count_no_persist / num_runs
p_Ao = count_A_only / num_runs
p_Bo = count_B_only / num_runs
p_AB = count_A_and_B / num_runs
p_A = p_Ao + p_AB
p_B = p_Bo + p_AB
p_any = 1.0 - p_no
p_diff = p_A - p_B

mean_A_unc = total_areas_A / num_runs
mean_B_unc = total_areas_B / num_runs
mean_A_cond = (total_areas_A_cond / count_runs_A_active) if count_runs_A_active else 0.0
mean_B_cond = (total_areas_B_cond / count_runs_B_active) if count_runs_B_active else 0.0

p_spont = count_spont_switch / num_runs
spont_json = json.dumps(spont_switch_details_list)

# Store aggregated stats
aggregated_results.append((
    G_var, alpha, j_max,
    p_no, p_Ao, p_Bo, p_AB,
    p_A, p_B, p_any, p_diff,
    mean_A_unc, mean_B_unc,
    mean_A_cond, mean_B_cond,
    p_spont, spont_json
))

# Area-frequency results
for i, area in enumerate(area_names):
    frac_A = area_count_A[i] / num_runs
    frac_B = area_count_B[i] / num_runs
    area_freq_results.append((G_var, alpha, j_max, area, frac_A, frac_B))

# Write aggregated CSV
pd.DataFrame(aggregated_results, columns=[
    "G","alpha","j_max",
    "Prob_NoPersist","Prob_A_Only","Prob_B_Only","Prob_A_B",
    "Prob_A","Prob_B","Prob_AnyPersist","ProbDiff(A-B)",
    "MeanAreasA_uncond","MeanAreasB_uncond",
    "MeanAreasA_cond","MeanAreasB_cond",
    "Prob_SpontSwitch","SpontSwitchDetails"
]).to_csv(os.path.join(data_dir, "persistent_activity_stats.csv"), index=False)

# Write area-frequency CSV
area_freq_cols = ["G","alpha","j_max","area_name","fraction_A","fraction_B"]
pd.DataFrame(area_freq_results, columns=area_freq_cols) \
  .to_csv(os.path.join(data_dir, "area_frequency_stats.csv"), index=False)

print(f"=== Simulation runs complete. All data saved in: {data_dir} ===")
