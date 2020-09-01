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
VaR(99.5%) based on Variance-Covariance for 20 day prediction
"""

import os
os.chdir('C:/Users/s157148/Documents/GitHub/TimeGAN')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, t

df = pd.read_csv("data/Master_EONIA.csv", sep=";")
eonia = np.array(df.iloc[1:,9])
df = df.iloc[:, 1:] # Remove the Date variable from the dataset
df.iloc[1:, 8] = np.diff(df['EONIA'])
df = df.iloc[1:, :]

def calibration(data):
    model = LinearRegression().fit(np.array(data.drop('EONIA', axis=1)),
                                   np.array(data['EONIA']).reshape(-1,1))

    return [model.intercept_, model.coef_]

def VaR(r_0, risk_factors, data, time, percentile=0.995, upward=True):
    # Calculate the parameters
    intercept, coef = calibration(data)
    
    # Calculate the covariance matrix
    cov_matrix = np.array(data.drop('EONIA', axis=1).cov())
    
    # Calculate the Value-at-Risk
    if upward:
        return np.ravel(r_0 + time * (intercept + np.matmul(np.transpose(np.ravel(coef)), risk_factors)) + \
               np.sqrt(time) * np.sqrt(np.matmul(np.matmul(np.transpose(np.ravel(coef)),
                                         cov_matrix), np.ravel(coef))) * t.ppf(percentile, 1.9))
    else:
        return np.ravel(r_0 + time * (intercept + np.matmul(np.transpose(np.ravel(coef)), risk_factors)) - \
               np.sqrt(time) * np.sqrt(np.matmul(np.matmul(np.transpose(np.ravel(coef)),
                                   cov_matrix), np.ravel(coef))) * t.ppf(percentile, 1.9))
            
# We have to scale the variance in order to get good performance

current_situation = np.array(df.drop('EONIA', axis=1).iloc[-1,:])
VaR(eonia[-1], current_situation, df, 20, 0.995, False)

# 4. Implement the Kupiec Test for the variance covariance simulation
upper = 0
lower = 0

for i in range(250):
    upper += eonia[3543+i] > VaR(eonia[3542+i], np.array(df.drop('EONIA', axis=1).iloc[3542+i]), df.iloc[2500+i:3542+i,:], 10, 0.995, True)
    lower += eonia[3543+i] < VaR(eonia[3542+i], np.array(df.drop('EONIA', axis=1).iloc[3542+i]), df.iloc[2500+i:3542+i,:], 10, 0.995, False)
    
# Simulate future scenarios and check the number of distinct neighbours
intercept, coef = calibration(df.iloc[2500:3542,:])
current_situation = np.array(df.drop('EONIA', axis=1).iloc[3542,:])

# Calculate the covariance matrix
cov_matrix = np.array(df.iloc[2500:3542,:].drop('EONIA', axis=1).cov())

new_eonia = np.zeros(shape=(2,250))
new_eonia[0, :] = eonia[3542]

for i in range(250):
    for j in range(1,2):
        new_eonia[j, i] = new_eonia[j-1, i] + intercept + np.sqrt(np.matmul(np.transpose(np.ravel(coef)), current_situation) + np.matmul(np.matmul(np.transpose(np.ravel(coef)),
                                       cov_matrix), np.ravel(coef))) * np.random.normal(0,1,1)
    
        current_situation = np.array(df.drop('EONIA', axis=1).iloc[3542+j,:])

simulated = np.diff(new_eonia, axis=0)

# 1. Import the EONIA data
data = pd.read_csv("data/EONIA.csv", sep=";")
data = np.array(data.iloc[:525,2])[::-1] # Filter for only EONIA 

# Check how much the differentials are similar
differentials = np.zeros(shape=(2,250))

# Make the real differential data set
for i in range(250):
    differentials[:,i] = data[250+i:252+i]
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
