from neuromaps.datasets import fetch_annotation
from neuromaps.parcellate import Parcellater
from neuromaps import transforms
from nilearn.datasets import fetch_atlas_schaefer_2018
from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# FC gradient

# Get annotation
fc_grad = fetch_annotation(source='margulies2016', desc='fcgradient01', space='fsLR', den='32k')
# Get Schaefer atlas and transform to fsLR space
schaefer = fetch_atlas_schaefer_2018(n_rois=200)
print(schaefer.maps)
schaefer_fslr = transforms.mni152_to_fslr(schaefer.maps, '32k', method='nearest')
# Parcellate data
parc = Parcellater(schaefer_fslr, 'fslr')
fc_grad_parc = parc.fit_transform(data=(fc_grad[0], fc_grad[1]), space='fslr', ignore_background_data=True)
fc_grad_parc_df = pd.DataFrame(fc_grad_parc, columns={'FC_grad': fc_grad})
# Save fc_grad to CSV
fc_grad_parc_df.to_csv('../../data/processed/schaefer200_fc_grad.csv', float_format='%.18f')

# Left hemisphere save to CSV
fc_grad_parc_lh = fc_grad_parc[:100]
fc_grad_parc_df = pd.DataFrame({'FC_grad': fc_grad_parc_lh})
fc_grad_parc_df.to_csv('../../data/processed/schaefer200_LH_fc_grad.csv')

# hierarchy proxy from FC gradient, same between 0 and 1 as t1t2-derived hierarchy
fc_hierarchy = (fc_grad_parc_lh - fc_grad_parc_lh.min()) / np.ptp(fc_grad_parc_lh)
fc_hierarchy_lh_df = pd.DataFrame({'FC_hierarchy_val': fc_hierarchy})
fc_hierarchy_lh_df.to_csv('../../data/processed/schaefer200_LH_FC_grad_hierarchy.csv')