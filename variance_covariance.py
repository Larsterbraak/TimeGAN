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
from scipy.stats import norm

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
               np.sqrt(time) * np.matmul(np.matmul(np.transpose(np.ravel(coef)),
                                         cov_matrix), np.ravel(coef)) * norm.ppf(percentile))
    else:
        return np.ravel(r_0 + time * (intercept + np.matmul(np.transpose(np.ravel(coef)), risk_factors)) - \
               np.sqrt(time) * np.matmul(np.matmul(np.transpose(np.ravel(coef)),
                                   cov_matrix), np.ravel(coef)) * norm.ppf(percentile))

current_situation = np.array(df.drop('EONIA', axis=1).iloc[-1,:])
VaR(eonia[-1], current_situation, df, 20, 0.995, False)

# 4. Implement the Kupiec Test for the variance covariance simulation
upper = 0
lower = 0

for i in range(250):
    upper += eonia[3562+i] > VaR(eonia[3542+i], np.array(df.drop('EONIA', axis=1).iloc[3542+i]), df.iloc[:3542+i, :], 20, 0.995, True)
    lower += eonia[3562+i] < VaR(eonia[3542+i], np.array(df.drop('EONIA', axis=1).iloc[3542+i]), df.iloc[:3542+i, :], 20, 0.995, False)