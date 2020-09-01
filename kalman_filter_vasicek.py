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
from scipy.stats import t

# 1. Import the EONIA data
data = pd.read_csv("data/EONIA.csv", sep=";")
data = np.array(data.iloc[:525,2])[::-1] # Filter for only EONIA 

t.fit(data)
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
        return np.ravel(expectation + np.sqrt(variance) * t.ppf(percentile,1.9))
    else:
        return np.ravel(expectation - np.sqrt(variance) * t.ppf(percentile,1.9))   

# 4. Implement the Kupiec test for Vasicek model
upward = 0
downward = 0

for i in range(250):
    upward += data[i+270] > VaR(data[i+250], data[i:i+250], 20, percentile=0.995, upward=True)
    downward += data[i+270] < VaR(data[i+250], data[i:i+250], 20, percentile=0.995, upward=False)
    
# 4. Number of simulated diverse nearest neighbours
# 4.1 We simulate using the Euler discritization
n = 250
t = 21
k, mu, sigma = model_calibration(data[:250])

r = np.zeros(shape=(t,n))
r[0,:] = data[250]
dr = np.zeros(shape=(t,n))
for j in range(n):
    for i in range(1,t):
        dr[i,j] = k*(mu - r[i-1,j]) + sigma * np.sqrt(1) * np.random.normal(loc=0, scale=1, size=1) #np.random.standard_t(1.9, 1) 
        r[i, j] = r[i-1, j] + dr[i,j]
        
simulated = dr[1:,:]
        
# Check how much the differentials are similar
differentials = np.zeros(shape=(21,250))

# Make the real differential data set
for i in range(250):
    differentials[:,i] = data[250+i:271+i]
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