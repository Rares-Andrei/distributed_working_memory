import nibabel.freesurfer as fs
from netneurotools.datasets import fetch_schaefer2018
import pandas as pd

schaefer_surface = fetch_schaefer2018()
labels_schaefer, ctab_schaefer, region_names_schaefer = fs.read_annot('../../data/raw/brain_annotations/Schaefer/fsaverage/atl-Schaefer2018_space-fsaverage_hemi-L_desc-200Parcels7Networks_deterministic.annot')

# Save region names Schaefer200 left hemisphere (the annot file was loaded for left hemisphere)
region_names_schaefer_decoded = [r.decode('utf-8') for r in region_names_schaefer][1:] # First region is Background+FreeSurfer_Defined_Medial_Wall
region_names_schaefer_decoded_df = pd.DataFrame({'Area_name' : region_names_schaefer_decoded})
region_names_schaefer_decoded_df.to_csv('../../data/processed/schaefer200_LH_area_names.csv')