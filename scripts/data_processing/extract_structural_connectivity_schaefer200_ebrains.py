# Script to extract structural connectivity matrix from data at https://search.kg.ebrains.eu/instances/3f179784-194d-4795-9d8d-301b524ca00a
# on Schaefer 200 parcellation
# Each row is a target node and each column a source node
import numpy as np
import pandas as pd
import glob

path = "../../data/raw/structural_connectivity_schaefer200_EBRAINS/*Counts.csv"
fiber_counts_files = []
# Get paths to all fiber counts files downloaded
for fname in glob.glob(path):
    fiber_counts_files.append(fname)
all_counts = []
# Add all data from files to all_counts
for count_file in fiber_counts_files:
    data = np.genfromtxt(count_file, delimiter=',')
    all_counts.append(data)
all_counts = np.array(all_counts)
print('Shape of all_counts:', all_counts.shape)
# Average across files
mean_counts = np.mean(all_counts, axis=0)
# Make elements on diagonal 0
for index in range(mean_counts.shape[0]):
    mean_counts[index, index] = 0.0
# Normalize between 0 and 1
mean_counts_normal = (mean_counts - np.min(mean_counts)) / (np.max(mean_counts) - np.min(mean_counts)) # This is only one way to normalize, could also just divide by max
print('Shape of normalized mean counts:', mean_counts_normal.shape)
df_mean_counts_normal = pd.DataFrame(mean_counts_normal)
df_mean_counts_normal.to_csv('../../data/processed/schaefer200_sc.csv', index=False)
mean_counts_normal_left_hemisphere = mean_counts_normal[:100, :100] # Left hemisphere
mean_counts_normal_left_hemisphere_df = pd.DataFrame(mean_counts_normal_left_hemisphere)
mean_counts_normal_left_hemisphere_df.to_csv('../../data/processed/schaefer200_LH_sc.csv', index=False)
