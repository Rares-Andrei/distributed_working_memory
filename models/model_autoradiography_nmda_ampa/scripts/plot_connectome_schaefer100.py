import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting

def plot_connectome(ax, adjacency_matrix, centroids_coords):
    plotting.plot_connectome(
    adjacency_matrix,
    centroids_coords,
    display_mode='l',
    node_color='black',
    edge_cmap='Greys',
    axes=ax
)

# Load adjacency & centroids
adjacency_matrix = pd.read_csv(
    '../../../data/processed/schaefer100_LH_sc.csv'
)
adjacency_matrix_numpy = adjacency_matrix.to_numpy()

centroids_rois = pd.read_csv(
    '../../../data/raw/Centroids_schaefer100/centroids_schaefer100_7networks.csv'
).iloc[:50]
centroids_coords = centroids_rois[['R','A','S']].to_numpy()

# Plot connectome (left hemisphere)
fig = plt.figure(figsize=(6,6), constrained_layout=True)
ax = fig.add_subplot(1,1,1)

plot_connectome(ax, adjacency_matrix_numpy, centroids_coords)

for ext in ['pdf']:
    fig.savefig(
        f'../results/connectome/connectome_schaefer100.{ext}',
        bbox_inches='tight'
    )