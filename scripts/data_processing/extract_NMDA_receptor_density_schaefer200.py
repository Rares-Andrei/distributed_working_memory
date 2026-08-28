"""
Based on Hansen et al. (2022)
Extracts NMDA receptor density map based on the Schaefer 200 parcellation.
"""
import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from neuromaps.parcellate import Parcellater
import pandas as pd

schaefer = fetch_atlas_schaefer_2018(n_rois=200)
receptors_nii = '../../data/raw/PET_neurotransmitters/NMDA_ge179_hc29_galovic.nii.gz'

parcellater = Parcellater(schaefer['maps'], 'MNI152')
schaefer_NMDA_rd = parcellater.fit_transform(receptors_nii, 'MNI152', True)
schaefer_NMDA_rd = schaefer_NMDA_rd[0]
schaefer_NMDA_rd_df = pd.DataFrame({'NMDA_rd': schaefer_NMDA_rd})
schaefer_NMDA_rd_df.to_csv('../../data/processed/schaefer200_NMDA_rd.csv', index=False)

schaefer_NMDA_rd_lh = schaefer_NMDA_rd[:100]
schaefer_NMDA_rd_lh_df = pd.DataFrame({'NMDA_rd': schaefer_NMDA_rd_lh})
schaefer_NMDA_rd_lh_df.to_csv('../../data/processed/schaefer200_LH_NMDA_rd.csv', index=False)