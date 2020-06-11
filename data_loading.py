"""
MSc Thesis Quantitative Finance
Title: Interest rate risk due to EONIA-ESTER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: May 25th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Data loading
(1) Load short rate dataset
 - Transform the raw data to preprocessed data

Inputs
(1) EONIA, pre-ESTER & ESTER dataset
- Raw data
- seq_length: Sequence Length

Outputs
- Preprocessed time series of European short rates
"""

import os

# Change to the needed working directory
os.chdir('C://Users/s157148/Documents/Github/TimeGAN')

# Necessary packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def create_dataset(name='pre-ESTER', normalization='min-max',
                   seq_length=20, training=True, multidimensional=True,
                   ester_probs=False, include_spread=False):
    
    if name == 'EONIA':
        df = pd.read_csv("data/EONIA.csv", sep=";")
        df = df.iloc[:, 2:] # Remove the Date variable from the dataset
        df = df.iloc[::-1] # Make dataset chronological
        df = np.ravel(np.diff(df, axis = 0))
        multidimensional = False
    elif name == 'pre-ESTER':
        df = pd.read_csv('data/pre_ESTER.csv', sep = ';')
        df = df.iloc[:, 1:] # Remove the Date variable from the dataset
        df = df.iloc[::-1] # Make dataset chronological
        # Make daily differencing
        df.iloc[1:,[1,2,4]] = np.diff(df[['R25', 'R75', 'WT']], axis =0)
        df = df.iloc[1:]
    elif name == 'ESTER':
        df = pd.read_csv("data/ESTER.csv", sep=";")
        df = df.iloc[:, 1:] # Remove the Date variable from the dataset
        df = df.iloc[::-1]# Make dataset chronological
        if include_spread:
            df = df + 0.0085
        # Make daily differencing
        df.iloc[1:,[1,2,4]] = np.diff(df[['R25', 'R75', 'WT']], axis =0)
        df = df.iloc[1:]
    else:
        return 'Non-existent dataset. Short rate data can be \
            "EONIA", "pre-ESTER" or "ESTER."'
    
    if not multidimensional and (name == 'pre-ESTER' or name =='ESTER'):
        df = np.array(df.WT)
    
    if normalization == 'min-max':
        if multidimensional:
            df = MinMaxScaler().fit_transform(df)
    else:
        return 'Still have to implement other normalization \
            techniques.'
            
    dataX = []
    
    # Make lookback periods of seq_length in the data
    for i in range(0, len(df) - seq_length):
        _df = df[i : i + seq_length]
        dataX.append(_df)
    
    # Create random permutations to make it more i.i.d.
    idx = np.random.permutation(len(dataX))
    if not ester_probs:         
        outputX = []
        for i in range(len(dataX)):
            outputX.append(dataX[idx[i]])
         
    if not multidimensional:
        # Reshape to be used by Tensorflow    
        outputX = np.reshape(dataX, newshape=(len(dataX), 
                                                seq_length, 
                                                1))
    else:
        # Reshape to be used by Tensorflow    
        outputX = np.reshape(outputX, newshape=(len(outputX), 
                                                seq_length, 
                                                df.shape[1]))
    
    if training:
        split = int(np.round(outputX.shape[0] * 2 / 3)) # Split in train & test
        X_train = outputX[0:split,:,:]
        X_test = outputX[split:,:,:]
        
        X_train = tf.data.Dataset.from_tensor_slices(tf.cast(X_train, tf.float64)).batch(50)
        X_test = tf.data.Dataset.from_tensor_slices(tf.cast(X_test, tf.float64)).batch(50)
        return X_train, X_test
    else:
        return idx, outputX

def rescale(df, N, seq_length, X_hat_scaled):
    if df == 'EONIA':
        df = pd.read_csv("data/EONIA.csv", sep=";")
        df = df.iloc[:, 1:] # Remove Date variable from dataset
        df = np.ravel(np.diff(df, axis = 0))
    elif df == 'pre-ESTER':
        df = pd.read_csv('data/pre_ESTER.csv', sep = ';')
        df = df.iloc[:, 1:] # Remove Date variable from dataset
        df = df.iloc[::-1] # Make dataset chronological
        # Make daily differencing
        df.iloc[1:,[1,2,4]] = np.diff(df[['R25', 'R75', 'WT']], axis =0)
        df = df.iloc[1:]
    elif df == 'ESTER':
        df = pd.read_csv("data/ESTER.csv", sep=";")
        df = df.iloc[:115, :]
    else:
        return 'Non-existent dataset. Short rate data can be \
            "EONIA", "pre-ESTER" or "ESTER."'
    
    # Rescale back to minimum
    minimum = np.reshape(np.array([np.min(df).values,]*N*seq_length),
                         (N, seq_length, df.shape[1]))

    maximum = np.reshape(np.array([np.max(df).values,]*N*seq_length), 
                         (N, seq_length, df.shape[1]))
    
    return X_hat_scaled * (maximum - minimum) + minimum
