"""
MSc Thesis Quantitative Finance
Title: Interest rate risk simulation using TimeGAN 
       after the EONIA-€STER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: August 19th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Inputs
Short rate time series

Outputs
VaR(99.5%) based on Kalman Filter for 20 day prediction
VaR(99.5%) based on Linear Regression estimation of 1-factor Vasicek for 20 day prediction
"""

import os
os.chdir('C:/Users/s157148/Documents/GitHub/TimeGAN')
    
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import norm

# 1. Import the EONIA data
data = pd.read_csv("data/Master_EONIA.csv", sep=";")
data = np.array(data.iloc[3547:, 9]) # Filter for only EONIA 

# 2. Vasicek model calibration using Linear Regression
def model_calibration(data):
    model = LinearRegression().fit(data[:-1].reshape(-1,1), data[1:].reshape(-1,1))
    errors = data[:-1] - (model.intercept_ + model.coef_ * data[1:])
    
    # Computation of model parameters
    k = -np.log(model.coef_)
    mu = model.intercept_ / (1 - model.coef_)
    sigma = np.std(errors) * np.sqrt((- 2 * np.log(model.coef_) ) / (1 - model.coef_**2))
    
    return [k, mu, sigma]

# 3. Value-at-Risk calculation
def VaR(r_0, data, time, percentile=0.995, upward=True):
    # Model calibration
    k, mu, sigma = model_calibration(data)
    
    # VaR calculation
    expectation = r_0 * np.exp(-k*time) + mu * (1 - np.exp(-k * time))
    variance = (sigma**2 / 2*k)*(1 - np.exp(-2*k*time))
    if upward:
        return np.ravel(expectation + np.sqrt(variance) * norm.ppf(percentile))
    else:
        return np.ravel(expectation - np.sqrt(variance) * norm.ppf(percentile))
    
VaR(-0.36, data, 20, True)
