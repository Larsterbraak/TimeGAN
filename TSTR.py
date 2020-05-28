"""
MSc Thesis Quantitative Finance
Title: Interest rate risk due to EONIA-ESTER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: May 25th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Train on Synthetic, Test on Real
(1) Simulate N trajectories of the short rate
 - Calculate Value-at-Risk at 99 percentile

Inputs
- Random seed
- Calibrated TimeGAN model

Outputs
- N simulations of short rate
- VaR(99%)
"""

import numpy as np
import matplotlib.pyplot as plt

def value_at_risk(X_hat, percentile = 99, upper = True):
    VaR = []
    
    for i in range(X_hat.shape[0]):
        _x = np.cumsum(X_hat[i, :, 0])
        VaR.append(_x[-1])
        
    VaR = np.array(VaR)
    
    if upper:
        return np.percentile(VaR, percentile)
    else:
        return np.percentile(VaR, 100-percentile)    