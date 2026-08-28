# Script that runs simulation at the optimal paramters for the main figures of the paper
# Outputs delay firing rates for optimal conditions

import os
import sys
import numpy as np

# Add the project root to the Python PATH
script_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.data import load_and_preprocess_data as get_data
from src.models import WorkingMemoryNetwork
from config.WM_params import WorkingMemoryParameters
import time
np.set_printoptions(threshold=np.inf)

random_seed = 666
outside_rng = np.random.default_rng(random_seed)

data = get_data(data_folder_path='../../../data/processed', parcellation='schaefer200')
area_names = data['area_names']
SLN = data['SLN']
W = data['W']
h = data['hier_vals']
normalized_nmda_rd = data['nmda_rd']

# Initial condition and parameters
N = len(area_names)
Y0 = np.zeros((N, 8))
Y0[:, 0:3] = 5 * (1 + np.tanh(outside_rng.normal(0, 2, (N, 3)))) # r values are initialized randomly between 0 and 10
WM_params = WorkingMemoryParameters()
WM_params.update(G=0.83, ALPHA=1.34, J_MAX=0.3) # Update G and alpha to optimal settings
# Generate the network
WM_network = WorkingMemoryNetwork(area_names=area_names, Y0=Y0, W=W, F=SLN, h=h,
                                  normalized_nmda_rd=normalized_nmda_rd,
                                  params=WM_params, gradient='NMDA')
# Run the simulation
t_end = 7
dt = 0.5e-3
stim_start_time = 2
stim_duration = 0.5
t_eval = np.arange(0, t_end+dt, dt) # Time points of the simulation
I_ext_strength = np.zeros((N, 3))

# Target names for which we want to assign 0.3
target_names = ['Vis_5', 'Vis_7', 'Vis_9']
# Find indexes of the target names in the list
rows = np.where(np.isin(area_names, target_names))[0]
I_ext_strength[rows, 0] = 0.3

start = time.perf_counter()
state_history, input_history = WM_network.run(t_end=t_end, dt=dt, I_ext_strengths=np.stack([I_ext_strength]),
                               ts_ext_start=np.array([stim_start_time]),
                               ts_ext_end=np.array([stim_start_time + stim_duration]))
end = time.perf_counter()
print("Elapsed (after compilation) = {}s".format((end - start)))

avg_firing_rates_A, avg_firing_rates_B = WM_network.save_avg_firing_rates_last_two_seconds(variables=['r_A', 'r_B'])
state_history_extract = WM_network.get_state_histories(['r_A'])[-1]
np.save('../results/main_task/cue_FR_G083_alpha134.npy', state_history_extract[int(stim_start_time / dt) + 100, :, 0])
avg_firing_rates = WM_network.save_avg_firing_rates_last_two_seconds()
np.save('../results/main_task/delay_FR_G083_alpha134.npy', avg_firing_rates)

_, pops = WM_network.check_sustained_activity_after_stim_removal()

# Need condition to check if it's A or B there, right now r_B is at index 0, it gets pops_A as pops_B.
try:
    pops_A = pops[0][1]
except:
    pops_A = []
try:
    pops_B = pops[1][1]
except:
    pops_B = []

print('Nodes where population B is persistent:', pops_B)
spont_switch, pop_switch = WM_network.check_spontaneous_switch_to_high_state(stim_start_time)
print(f'There is spontaneous switch to higher state: {spont_switch}, in populations {pop_switch}')
changes_detected, changed_populations = WM_network.check_sustained_activity_after_stim_removal()
print(changed_populations)
try:
    print('Number of persistent areas:', len(changed_populations[0][1]))
    areas_persistent = [area_names[i] for i in changed_populations[0][1]]
    print('Areas persistent:', areas_persistent)
    np.save('../results/main_task/persistent_areas_G083_alpha134.npy', areas_persistent)
except:
    print('No persistent areas')

# Plot the results
area_names_for_figure = ['Early Visual Cortex 1', 'Early Visual Cortex 2', 'Area p47r',
                         '9m', '6ma', 'PGi',
                         'Area 46', 'Area 9-46v']
_ = WM_network.plot_n_areas(xlim=(-2, t_end - 2), ylim=(-5, 65),
                            target_area_names=['Vis_9', 'Vis_5', 'Cont_PFCl_1', 'Default_PFC_7',
                                               'DorsAttn_FEF_2', 'Default_Par_4', 'SalVentAttn_PFCl_1',
                                               'Cont_PFCl_3'],
                            names_for_figures=area_names_for_figure,
                            variables=['r_A', 'r_B'], colors=['#56c68a', '#eaa8b7'],
                            save_path='../results/main_task/persistent_activity.svg',
                            legend=False,
                            show=False)
# _ = WM_network.plot_all_areas(xlim=(-2, t_end - 2),
#                             variables=['r_A', 'r_B'], colors=['#56c68a', '#eaa8b7'],
#                             save_path='../results/main_task/persistent_activity_all_areas.svg',
#                             legend=False,
#                             show=False)
highlight_areas = ['Vis_10', 'SomMot_2', 'SalVentAttn_PFCl_1', 'Default_Temp_5', 'SomMot_6', 'Cont_PFCl_3', 'Default_PFC_7', 'Vis_14']
_ = WM_network.plot_n_areas(xlim=(-2, t_end - 2), ylim=(-2, 25),
                            target_area_names=highlight_areas,
                            names_for_figures=highlight_areas,
                            variables=['r_A', 'r_B'], colors=['#56c68a', '#eaa8b7'],
                            save_path='../results/main_task/traces_select_areas.svg',
                            legend=False,
                            show=False)
_ = WM_network.plot_avg_firing_rates_last_two_seconds(save_path='../results/main_task/avg_last_2_s.svg',
                                                       color=['#56c68a', '#eaa8b7'], show=False)
# rates_A, rates_B = WM_network.save_avg_firing_rates_last_two_seconds(variables=['r_A', 'r_B'])
# i = list(area_names).index('Vis_10')
# print('Vis_10 -> r_A =', rates_A[i], ' r_B =', rates_B[i])