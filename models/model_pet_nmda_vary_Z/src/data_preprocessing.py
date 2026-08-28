import numpy as np

def process_SC(SC: np.ndarray) -> np.ndarray:
    """
    Process structural connectivity (SC) values to compute the weight matrix W.
    
    Calculates: W = 1.2 * SC**0.3 normalized by the sum along each row.
    """
    W = 1.2 * np.power(SC, 0.3)
    # Normalize each row so the weights sum to 1
    W /= np.sum(W, axis=1, keepdims=True)
    return W

def normalize_nmda_rd(nmda_rd: np.ndarray) -> np.ndarray:
    """
    Normalize NMDA receptor density values between 0 and 1.
    """
    return (nmda_rd - nmda_rd.min()) / (nmda_rd.max() - nmda_rd.min())

def process_area_names(area_names: np.ndarray) -> list:
    """
    Remove unnecessary prefixes from area names.
    Valid in the case of using the original Schaefer200 parcel names.
    If using a different parcellation, ignore.
    """
    return [area.replace('7Networks_LH_', '') for area in area_names]