"""
MSc Thesis Quantitative Finance
Title: Interest rate risk simulation using TimeGAN 
       after the EONIA-€STER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: October 23rd 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Inputs
Short rate time series

Outputs
VaR(99%) based on Linear Regression estimation of 1-factor Vasicek for 20 day prediction
"""

import os
os.chdir('C:/Users/s157148/Documents/GitHub/TimeGAN')
    
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import t as tdistr

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
def VaR(r_0, data, time, percentile=0.99, upward=True):
    # Model calibration
    k, mu, sigma = model_calibration(data)
    
    # VaR calculation
    expectation = r_0 * np.exp(-k*time) + mu * (1 - np.exp(-k * time))
    variance = (sigma**2 / 2*k)*(1 - np.exp(-2*k*time))
    
    # Estimate the degrees of freedom for the t-distribution
    t_df = tdistr.fit(data)[0]
    
    if upward:
        return np.ravel(expectation + np.sqrt(variance) * tdistr.ppf(percentile, t_df))
    else:
        return np.ravel(expectation - np.sqrt(variance) * tdistr.ppf(percentile, t_df))   

# 1. Import the EONIA data for validation error
data = pd.read_csv("data/EONIA.csv", sep=";")
data = np.array(data.iloc[:775,2])[::-1] # Filter for only EONIA 

# 3. Implement the Kupiec test for Vasicek model during the training data
upward = 0
downward = 0

# Loop over 250 trading days in the training data set
for i in range(250):
    upward += data[i+370] > VaR(data[i+250], data[i:250+i], 120, percentile=0.99, upward=True)
    downward += data[i+370] < VaR(data[i+250], data[i:250+i], 120, percentile=0.99, upward=False)

# 1. Import the EONIA data for test error
data = pd.read_csv("data/EONIA.csv", sep=";")
data = np.array(data.iloc[:525,2])[::-1] # Filter for only EONIA 

# 4. Implement the Kupiec test for Vasicek model during the test data
upward = 0
downward = 0

# Loop over 250 trading days in the test data set
for i in range(250):
    upward += data[i+270] > VaR(data[i+269], data[i:i+269], 1, percentile=0.99, upward=True)
    downward += data[i+270] < VaR(data[i+269], data[i:i+269], 1, percentile=0.99, upward=False)
    
# 4. Number of simulated diverse nearest neighbours
# 4.1 We simulate using the Euler discritization
n = 250 # simulate n times
t = 20 # simulate T days
k, mu, sigma = model_calibration(data[:250]) # calibrate the model 

r = np.zeros(shape=(t+1,n))
r[0,:] = data[250]
dr = np.zeros(shape=(t+1,n))
for j in range(n):
    for i in range(1,t+1):
        dr[i,j] = k*(mu - r[i-1,j]) + sigma * np.sqrt(1) * np.random.normal(loc=0, scale=1, size=1) #np.random.standard_t(1.9, 1) 
        r[i,j] = r[i-1, j] + dr[i,j]
        
simulated = dr[1:,:]
        
# Check how much the differentials are similar
differentials = np.zeros(shape=(21,250))

# Make the real differential data set
for i in range(250):
    differentials[:,i] = data[(250+i):(271+i)]
real = np.diff(differentials, axis = 0)

# Check the number of distinct neighbours
neighbours = np.zeros(shape=(250,))
for i in range(250):
    number=0
    best = np.mean((simulated[:,i] - real[:,number])**2)
    for j in range(1, 250):
        if np.mean((simulated[:,i] -  real[:, j])**2) < best:
            best = np.mean((simulated[:,i] -  real[:, j])**2)
            number = j
    neighbours[i] = number        
    
uniques = np.unique(neighbours)                