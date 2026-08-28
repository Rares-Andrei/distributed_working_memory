import os
import bct

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

from .data_io import (
    load_area_names,
    load_SLN,
    load_SC,
    load_hierarchy_values,
    load_nmda_rd
)
from .data_preprocessing import process_SC, normalize_nmda_rd, process_area_names

def load_and_preprocess_data(data_folder_path: str = f"{project_root}/data/processed",
                             parcellation: str = 'schaefer200', shuffled_SLN=False) -> dict:
    """
    Load and further further process the data for model input.
    
    Returns:
        dict: Contains keys 'area_names', 'SLN', 'W', 'hier_vals', and 'nmda_rd'
    """
    # Construct file paths
    areas_file = os.path.join(data_folder_path, f'{parcellation}_LH_area_names.csv')
    if shuffled_SLN:
        sln_file = os.path.join(data_folder_path, f'randomly_shuffled_{parcellation}_LH_SLN.csv')
    else:
        sln_file = os.path.join(data_folder_path, f'{parcellation}_LH_SLN.csv')
    sc_file = os.path.join(data_folder_path, f'{parcellation}_LH_sc.csv')
    hier_file = os.path.join(data_folder_path, f'{parcellation}_LH_hierarchy.csv')
    nmda_rd_file = os.path.join(data_folder_path, f'{parcellation}_LH_NMDA_rd.csv')
    
    # Loading raw data from CSVs (which are already processed from another data processing stage)
    area_names = load_area_names(areas_file)
    area_names_label_only = process_area_names(area_names)
    
    SLN = load_SLN(sln_file)
    
    W_pre = load_SC(sc_file)
    # Apply range compression and normalization
    W = process_SC(W_pre)
    
    hier_vals = load_hierarchy_values(hier_file)
    
    nmda_rd = load_nmda_rd(nmda_rd_file)
    normalized_nmda_rd = normalize_nmda_rd(nmda_rd)
    
    return {
        'area_names': area_names_label_only,
        'SLN': SLN,
        'W': W,
        'hier_vals': hier_vals,
        'nmda_rd': normalized_nmda_rd,
    }
