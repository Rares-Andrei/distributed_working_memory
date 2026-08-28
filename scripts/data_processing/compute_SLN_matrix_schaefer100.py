import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Left hemisphere SLN computed from LH hierarchies
h = pd.read_csv('../../data/processed/schaefer100_LH_hierarchy.csv')['Hierarchy_val']
# Based on Qingkai Zhu's Feedforward and Feedback Frequency-dependent Interplays in a Large-scale Laminar Model of the Human Cortex
# Master thesis formula: ldiffSLN
hierVal = h - np.min(h)
hierVal = hierVal / np.max(hierVal)
maxDiff = np.ptp(hierVal)
ldiffSLN = np.zeros((len(hierVal), len(hierVal)))
# i is the target, j is the source
# if target is very low in the hierarchy, the SLN will be close to 0 (small) for that target from a source that is high in hierarchy
for i in range(len(hierVal)):
    for j in range(len(hierVal)):
        ldiffSLN[i, j] = ((hierVal[i] - hierVal[j]) / (2 * maxDiff)) + 0.5

ldiffSLN -= np.diag(np.diag(ldiffSLN)) # Make diagonal elements equal to 0 (there are other self excitatory variables)
ldiffSLN_df = pd.DataFrame(ldiffSLN)
ldiffSLN_df.to_csv('../../data/processed/schaefer100_LH_SLN.csv', index = False)