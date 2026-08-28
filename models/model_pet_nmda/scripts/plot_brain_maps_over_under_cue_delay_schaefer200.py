import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from nibabel.freesurfer.io import read_annot
import nibabel.freesurfer as fs
from matplotlib.colors import LinearSegmentedColormap
cmaps = {
    "excitation_seq": LinearSegmentedColormap.from_list(
        "exc_seq", ["#e7f6ed", "#86c3a2", "#56c68a"], N=256
    )
}


def plot_multiple_surfaces(
    file_paths,
    annot_file_path,
    cmap=cmaps['excitation_seq'],
    save_dir=None,
    views=('lateral', 'medial'),
    dpi=1200,
    figsize=(6, 6),
    symmetric_cbar=False,
):
    """
    Plot surface stat maps for multiple datasets, with 'over_active' on its own color scale
    and the other two sharing a separate common color scale.

    Parameters
    ----------
    file_paths : dict
        Mapping from label (str) to .npy file path containing average FR values per region.
        Must contain 'over_active', 'optimal', and 'sub_active'.
    annot_file_path : str
        Full path to the Schaefer annotation .annot file for the left hemisphere.
    cmap : str, optional
        Matplotlib colormap name. Default 'plasma'.
    save_dir : str, optional
        Directory where to save output images. If None, saves in current directory.
    views : tuple of str
        Surface views to plot (e.g., ('lateral', 'medial')).
    dpi : int, optional
        Dots per inch for saved figures. Default 600.
    figsize : tuple, optional
        Figure size in inches. Default (6, 6).
    symmetric_cbar : bool, optional
        Whether to center the colorbar at zero. Default False.

    Returns
    -------
    None
    """
    # Read annotation
    labels, ctab, region_names = read_annot(annot_file_path)

    # Fetch fsaverage surfaces
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

    # Ensure expected labels
    if 'over_active' not in file_paths:
        raise KeyError("file_paths must include 'over_active'.")

    # Compute separate color scales
    # Overactive dataset alone
    arr_over = np.load(file_paths['over_active'])
    vmin_over, vmax_over = np.nanmin(arr_over), np.nanmax(arr_over)
    # Optimal + sub_active together
    other_keys = [k for k in file_paths if k != 'over_active']
    arr_others = np.concatenate([np.load(file_paths[k]) for k in other_keys])
    vmin_others, vmax_others = np.nanmin(arr_others), np.nanmax(arr_others)

    # Create save directory if needed
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Plot each dataset
    for label, path in file_paths.items():
        data_vals = np.load(path)
        # Choose color limits
        if label == 'over_active':
            vmin, vmax = 0, 25
        else:
            vmin, vmax = 0, 25

        # Map data to vertices
        mapped = np.full(labels.shape, np.nan, dtype=float)
        for idx in range(1, len(region_names)):
            mapped[labels == idx] = data_vals[idx - 1]

        # Plot for each view
        for view in views:
            fig = plt.figure(constrained_layout=True, figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            plotting.plot_surf_stat_map(
                fsaverage.infl_left,
                mapped,
                fsaverage.sulc_left,
                hemi='left',
                view=view,
                cmap=cmap,
                colorbar=True,
                symmetric_cbar=symmetric_cbar,
                vmin=vmin,
                vmax=vmax,
                axes=ax,
                figure=fig,
                engine='matplotlib',
                avg_method='median'
            )

            # Save figure
            fname = f"{label}_{view}.pdf"
            out_path = os.path.join(save_dir or '.', fname)
            fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

            
file_paths = {'over_active': '../results/over_under_active/delay_FR_G13_alpha025.npy',
              'optimal': '../results/main_task/delay_FR_G083_alpha134.npy',
              'sub_active': '../results/over_under_active/delay_FR_G06_alpha14.npy'}
save_dir = '../results/over_under_active'
annot_file_path = '../../../data/raw/brain_annotations/Schaefer/fsaverage5/atl-Schaefer2018_space-fsaverage5_hemi-L_desc-200Parcels7Networks_deterministic.annot'
plot_multiple_surfaces(file_paths=file_paths, annot_file_path=annot_file_path, save_dir=save_dir)