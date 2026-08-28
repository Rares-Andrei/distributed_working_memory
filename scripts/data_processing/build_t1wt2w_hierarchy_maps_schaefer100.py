from neuromaps.datasets import fetch_annotation
from neuromaps.parcellate import Parcellater
from neuromaps import transforms
from nilearn.datasets import fetch_atlas_schaefer_2018
from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Compute T1w:T2w map

# Get annotation
t1t2 = fetch_annotation(source='hcps1200', desc='myelinmap', space='fsLR', den='32k')
# Get Schaefer atlas and transform to fsLR space
schaefer = fetch_atlas_schaefer_2018(n_rois=100)
schaefer_fslr = transforms.mni152_to_fslr(schaefer.maps, '32k', method='nearest')
# Parcellate data
parc = Parcellater(schaefer_fslr, 'fslr')
t1t2_parc = parc.fit_transform(data=(t1t2[0], t1t2[1]), space='fslr', ignore_background_data=True)
t1t2_parc_df = pd.DataFrame(t1t2_parc, columns={'T1w:T2w': t1t2_parc})
# Save T1w:T2w to CSV
t1t2_parc_df.to_csv('../../data/processed/schaefer100_t1wt2w.csv', float_format='%.18f')

# Compute hierarchy based on T1w:T2w map

# Function from Demirtas et al (2019).
def linearize_map(x):
    """
    linearization function for T1w/T2w maps
    """
    return erf((x - np.mean(x)) / x.std() / np.sqrt(2))

# Apply linearization and normalize (ptp: peak-to-peak)
t1t2_lh_lin = linearize_map(t1t2_parc)
hierarchy = (np.max(t1t2_lh_lin) - t1t2_lh_lin) / np.ptp(t1t2_lh_lin)
# Save to CSV
hierarchy_df = pd.DataFrame({'Hierarchy_val': hierarchy})
hierarchy_df.to_csv('../../data/processed/schaefer100_hierarchy.csv')
# Left hemisphere save to CSV (normalzie before)
hierarchy_lh = hierarchy[:50]
hierarchy_lh_norm = (hierarchy_lh - np.min(hierarchy_lh)) / (np.max(hierarchy_lh) - np.min(hierarchy_lh))
hierarchy_lh_df = pd.DataFrame({'Hierarchy_val': hierarchy_lh_norm})
hierarchy_lh_df.to_csv('../../data/processed/schaefer100_LH_hierarchy.csv')