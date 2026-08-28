import pandas as pd
import numpy as np

def load_csv(file_path: str, header='infer', dtype=None) -> pd.DataFrame:
    """
    Generic CSV loader that wraps around pandas.read_csv.
    """
    return pd.read_csv(file_path, header=header, dtype=dtype)

def load_area_names(file_path: str) -> np.ndarray:
    """
    Load area names from a CSV file.
    """
    df = load_csv(file_path, header=0)
    area_names = df['Area_name'].to_numpy()
    return area_names

def load_SLN(file_path: str) -> np.ndarray:
    """
    Load hierarchy-inferred SLN from a CSV file.
    """
    df = load_csv(file_path)
    return df.to_numpy()

def load_SC(file_path: str) -> np.ndarray:
    """
    Load the structural connectivity (SC) matrix (DTI-based) from a CSV file.
    """
    df = load_csv(file_path)
    return df.to_numpy()

def load_hierarchy_values(file_path: str) -> np.ndarray:
    """
    Load hierarchy values from a CSV file.
    """
    df = load_csv(file_path, header=0)
    return df['Hierarchy_val'].to_numpy().flatten()

def load_nmda_rd(file_path: str) -> np.ndarray:
    """
    Load NMDA receptor density values from a CSV file.
    """
    df = load_csv(file_path, dtype=float)
    return df['NMDA_rd'].to_numpy().flatten()
