"""
MSc Thesis Quantitative Finance
Title: Interest rate risk due to EONIA-ESTER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: May 25th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Create simulation for the EONIA short rate and backtest the VaR
using the Basel Committee's Traffic Light coverage test   
(1) Perform coverage test Basel
 -
(2) Perform realness classification of ESTER + 8.5 bps wrt EONIA

Inputs
(1) EONIA, calibrated TimeGAN models
-
- 

Outputs
- Classification for the Value-at-Risk model
"""

import numpy as np
from TSTR import value_at_risk
from data_loading import create_dataset
from scipy.stats import binom
from scipy.special import expit
from training import RandomGenerator
    
def coverage_test_basel(generator_model, recovery_model,
                        lower=True, hidden_dim = 4):
    
    # Get the EONIA T-day real values
    EONIA = create_dataset(name='EONIA', seq_length = 20, training=False)
    EONIA = np.reshape(EONIA, (EONIA.shape[0], EONIA.shape[1])) # These T-day intervals are shuffled
        
    # Specify Nr. simulations and T
    N = 1000
    T = 20
    exceedances = 0 # Initialize the number of exceedances
    
    for i in range(250): # 250 trading days
        # N simulations for TimeGAN calibrated on EONIA
        Z_mb = RandomGenerator(N, [T, hidden_dim])
        X_hat_scaled = recovery_model(generator_model(Z_mb)).numpy()
        value = np.cumsum(EONIA[i, :])[-1] # Value for T-days
        
        # Train on Synthetic, Test on Real
        if lower:
            VaR_l = value_at_risk(X_hat=X_hat_scaled, percentile=99, upper=False)
            exceedances = exceedances + int(value < VaR_l)
        else:    
            VaR_u = value_at_risk(X_hat=X_hat_scaled, percentile=99, upper=True)
            exceedances = exceedances + int(VaR_u < value)
        
    prob = binom.cdf(np.sum(exceedances), 250, .01)
    
    if prob <= binom.cdf(4, 250, 0.01):
        return 'Green'
    elif binom.cdf(4, 250, 0.01) <= prob <= binom.cdf(9, 250, 0.01):
        return 'Yellow'
    else:
        return 'Red'   
    
def ester_classifier(embedder_model,
                     discriminator_model):
    ESTER = create_dataset(name='pre-ESTER', normalization='min-max',
                           seq_length=20, training=False, 
                           multidimensional=False, ester_probs=True)
   
    # Probably not necessary and does not work
    #ESTER = np.reshape(ESTER, (ESTER.shape[0], ESTER.shape[1])) # These T-day intervals are shuffled
    
    # Calculate the probabilities
    logistic_probs = discriminator_model(embedder_model(ESTER)).numpy()
    probs = expit(logistic_probs) # Revert back to [0,1] probabilities
    
    return probs
