# Script to combine structural connectivity and SLN to
# understand which nodes receive more excitation vs. inhibition

import numpy as np
import pandas as pd

W = np.array(pd.read_csv('../../data/processed/schaefer100_LH_sc.csv'))
SLN = np.array(pd.read_csv('../../data/processed/schaefer100_LH_SLN.csv'))

W_E = W * SLN
W_I = W * (1 - SLN)

np.save('../../data/processed/schaefer100_W_E_SC_SLN.npy', W_E)
np.save('../../data/processed/schaefer100_W_I_SC_SLN.npy', W_I)