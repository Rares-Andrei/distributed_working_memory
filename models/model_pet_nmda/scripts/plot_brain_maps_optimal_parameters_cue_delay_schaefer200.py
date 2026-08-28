# Script to plot brain maps during the cue and delay period under optimal parameters
# for model built with nmda PET imaging
import numpy as np
import matplotlib.pyplot as plt
from nibabel.freesurfer.io import read_annot
from nilearn import datasets, plotting
import nibabel.freesurfer as fs
import scienceplots
plt.style.use(['science', 'nature'])
from matplotlib.colors import LinearSegmentedColormap
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
cmaps = {
    "excitation_seq": LinearSegmentedColormap.from_list(
        "exc_seq", ["#e7f6ed", "#86c3a2", "#56c68a"], N=256
    )
}
cue_FR = '../results/main_task/cue_FR_G083_alpha134.npy' # Path to npy file containing the FR of brain areas at cue time (before persistent activation)
cue_FR = np.load(cue_FR)

labels_schaefer, ctab_schaefer, region_names_schaefer = fs.read_annot('../../../data/raw/brain_annotations/Schaefer/fsaverage/atl-Schaefer2018_space-fsaverage_hemi-L_desc-200Parcels7Networks_deterministic.annot')
region_names_schaefer = [r.decode('utf-8') for r in region_names_schaefer]
mapped_data_schaefer_cue = np.zeros_like(labels_schaefer, dtype=float) # Define a map with the same number of vertices as the annotation file
for i, region in enumerate(region_names_schaefer):
    region_idx = i
    if i == 0:
        mapped_data_schaefer_cue[labels_schaefer == region_idx] = None
    else:
        mapped_data_schaefer_cue[labels_schaefer == region_idx] = cue_FR[region_idx - 1] # First index in region_names is background
vmin, vmax = np.nanmin(mapped_data_schaefer_cue), np.nanmax(mapped_data_schaefer_cue)

fs = datasets.fetch_surf_fsaverage('fsaverage')
for view in ['lateral','medial']:
    fig = plt.figure(constrained_layout=True, figsize=(6,6))
    ax = fig.add_subplot(1,1,1, projection='3d')
    plotting.plot_surf_stat_map(
        fs.infl_left, mapped_data_schaefer_cue, fs.sulc_left,
        hemi='left', view=view,
        cmap=cmaps['excitation_seq'], colorbar=True,
        symmetric_cbar=False,
        vmin=0, vmax=vmax,
        axes=ax, figure=fig,
        engine='matplotlib', avg_method='median'
    )
    for ext in ['pdf']:
        dpi = 150 if ext in ['png'] else None
        fig.savefig(f'../results/main_task/cue_activity_{view}.{ext}', bbox_inches='tight')
    plt.close()
    
path_avg_FR_persistent ='../results/main_task/delay_FR_G083_alpha134.npy' # Path to npy file containing the average FR over last 2s of simulation
avg_FR_persistent = np.load(path_avg_FR_persistent)
mapped_data_schaefer_persistent = np.zeros_like(labels_schaefer, dtype=float) # Define a map with the same number of vertices as the annotation file
for i, region in enumerate(region_names_schaefer):
    region_idx = i
    if i == 0:
        mapped_data_schaefer_persistent[labels_schaefer == region_idx] = None
    else:
        mapped_data_schaefer_persistent[labels_schaefer == region_idx] = avg_FR_persistent[region_idx - 1] # 
fsaverage = datasets.fetch_surf_fsaverage('fsaverage') # Extract surface to plot on
vmin, vmax = np.nanmin(mapped_data_schaefer_persistent), np.nanmax(mapped_data_schaefer_persistent)

fs = datasets.fetch_surf_fsaverage('fsaverage')
for view in ['lateral','medial']:
    fig = plt.figure(constrained_layout=True, figsize=(6,6))
    ax = fig.add_subplot(1,1,1, projection='3d')
    plotting.plot_surf_stat_map(
        fs.infl_left, mapped_data_schaefer_persistent, fs.sulc_left,
        hemi='left', view=view,
        cmap=cmaps['excitation_seq'], colorbar=True,
        symmetric_cbar=False,
        vmin=0, vmax=vmax,
        axes=ax, figure=fig,
        engine='matplotlib', avg_method='median'
    )
    for ext in ['pdf']:
        dpi = 150 if ext in ['png'] else None
        fig.savefig(f'../results/main_task/persistent_activity_{view}.{ext}', bbox_inches='tight')
    plt.close()