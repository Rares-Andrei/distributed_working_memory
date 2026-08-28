# Script to extract structural connectivity matrix from data at https://search.kg.ebrains.eu/instances/3f179784-194d-4795-9d8d-301b524ca00a
# on Schaefer 200 parcellation
# Each row is a target node and each column a source node
import numpy as np
import pandas as pd
import glob

path = "../../data/raw/structural_connectivity_schaefer200_EBRAINS/*Lengths.csv"
fiber_lengths_files = []
# Get paths to all fiber lengths files downloaded
for fname in glob.glob(path):
    fiber_lengths_files.append(fname)
all_lengths = []
# Add all data from files to all_lengths
for lengths_file in fiber_lengths_files:
    data = np.genfromtxt(lengths_file, delimiter=',')
    all_lengths.append(data)
all_lengths = np.array(all_lengths)
print('Shape of all_lengths:', all_lengths.shape)
# Average across files
mean_lengths = np.mean(all_lengths, axis=0)
# Make elements on diagonal 0
for index in range(mean_lengths.shape[0]):
    mean_lengths[index, index] = 0.0
df_mean_lengths = pd.DataFrame(mean_lengths)
df_mean_lengths.to_csv('../../data/processed/schaefer200_lengths.csv', index=False)
mean_lengths_left_hemisphere = mean_lengths[:100, :100] # Left hemisphere
mean_lengths_left_hemisphere_df = pd.DataFrame(mean_lengths_left_hemisphere)
mean_lengths_left_hemisphere_df.to_csv('../../data/processed/schaefer200_LH_lengths.csv', index=False)

mean_delay = 20  # ms
conduction_speed = np.nanmean(mean_lengths_left_hemisphere[mean_lengths_left_hemisphere > 0]) / mean_delay # mm / ms = m/s
mean_delays_left_hemisphere = (1e-3 * mean_lengths_left_hemisphere) / conduction_speed  # s
print(f'mean length: {np.mean(mean_lengths_left_hemisphere)}')
print(f'conduction speed for a mean delay of 5ms: {conduction_speed} m/s')
print(f'mean delay: {np.mean(mean_delays_left_hemisphere[mean_delays_left_hemisphere > 0])}')
mean_delays_left_hemisphere_df = pd.DataFrame(mean_delays_left_hemisphere)
mean_delays_left_hemisphere_df.to_csv('../../data/processed/schaefer200_LH_delays.csv', index=False)
