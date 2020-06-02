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

import os
import numpy as np
from TSTR import value_at_risk
from data_loading import create_dataset
from scipy.stats import binom
from scipy.special import expit
from training import RandomGenerator
import matplotlib.pyplot as plt

# Change to the needed working directory
os.chdir('C://Users/s157148/Documents/Github/TimeGAN')

def load_models(epoch):        
    from models.Discriminator import Discriminator
    from models.Recovery import Recovery
    from models.Generator import Generator
    from models.Embedder import Embedder
    from models.Supervisor import Supervisor
     
    if epoch % 50 != 0:
        return 'Only insert epochs that are divisible by 50.'
    else:
        # Only use when you want to load the models
        e_model_pre_trained = Embedder('logs/e_model_pre_train', [], dimensionality = 1)
        e_model_pre_trained.load_weights('C://Users/s157148/Documents/Github/TimeGAN/weights/embedder/epoch_' + str(epoch)).expect_partial()
        e_model_pre_trained.build([])
        
        r_model_pre_trained = Recovery('logs/r_model_pre_train', [], dimensionality = 1)
        r_model_pre_trained.load_weights('C://Users/s157148/Documents/Github/TimeGAN/weights/recovery/epoch_' + str(epoch)).expect_partial()
        r_model_pre_trained.build([])
        
        s_model_pre_trained = Supervisor('logs/s_model_pre_train', [])
        s_model_pre_trained.load_weights('C://Users/s157148/Documents/Github/TimeGAN/weights/supervisor/epoch_' + str(epoch)).expect_partial()
        s_model_pre_trained.build([])
        
        g_model_pre_trained = Generator('logs/g_model_pre_train', [])
        g_model_pre_trained.load_weights('C://Users/s157148/Documents/Github/TimeGAN/weights/generator/epoch_' + str(epoch)).expect_partial()
        g_model_pre_trained.build([])
        
        d_model_pre_trained = Discriminator('logs/d_model_pre_train', []) 
        d_model_pre_trained.load_weights('C://Users/s157148/Documents/Github/TimeGAN/weights/discriminator/epoch_' + str(epoch)).expect_partial()
        d_model_pre_trained.build([])
        
        return e_model_pre_trained, r_model_pre_trained, s_model_pre_trained, g_model_pre_trained, d_model_pre_trained

def create_plot_simu(simulation_cum, T, ax):
    ax.plot(range(T), np.transpose(simulation_cum))
    ax.set_xlabel('Days')
    ax.set_ylabel('Short rate')

def create_plot_nn(equivalent, simulation, T, ax):
    ax.plot(range(T), equivalent, label = 'EONIA data')
    ax.plot(range(T), simulation, label = 'Generated data')
    ax.set_xlabel('Days')
    ax.set_ylabel('Short rate')

def image_grid(N, T, hidden_dim, recovery_model, generator_model):
    # Get the EONIA T-day real values
    EONIA = create_dataset(name='EONIA', seq_length = 20, training=False)
    EONIA = np.reshape(EONIA, (EONIA.shape[0], EONIA.shape[1])) # These T-day intervals are shuffled
        
    figure = plt.figure(figsize=(15,15))
    plt.title('Nearest neighbour in the EONIA data')
    for i in range(20):
        number = np.random.randint(N) # Simulate a random numbers
        
        # N simulations for TimeGAN calibrated on EONIA
        Z_mb = RandomGenerator(N, [T, hidden_dim])
        X_hat_scaled = recovery_model(generator_model(Z_mb)).numpy()
        simulation = np.reshape(X_hat_scaled, (N, T))
        simulation_cum = np.cumsum(simulation, axis=1)
        
        # Find the nearest neighbour in the dataset
        closest = np.mean(((EONIA - simulation[number, :])**2), axis = 1).argsort()[0]
        equivalent = EONIA[closest, :]
        
        # Start next subplot.
        if i < 10:
            ax = plt.subplot(4, 5, i + 1, title=str('Simulation ' + str(i+1)))
            create_plot_simu(simulation_cum, T, ax)
        else:
            ax = plt.subplot(4, 5, i + 1, title=str('Nearest Neighbour '+ str(i-9) ) )
            create_plot_nn(equivalent, simulation[number,:], T, ax)
        plt.grid(False)
    
    plt.tight_layout()
    return figure
 
def coverage_test_basel(generator_model, recovery_model,
                        lower=True, hidden_dim = 4):
    
    # Get the EONIA T-day real values
    EONIA = create_dataset(name='EONIA', seq_length = 20, training=False)
    EONIA = np.reshape(EONIA, (EONIA.shape[0], EONIA.shape[1])) # These T-day intervals are shuffled
        
    # Specify Nr. simulations and T
    N = 100
    T = 20
    exceedances = 0 # Initialize the number of exceedances
    
    for i in range(250): # 250 trading days
        # N simulations for TimeGAN calibrated on EONIA
        Z_mb = RandomGenerator(N, [T, hidden_dim])
        X_hat_scaled = recovery_model(generator_model(Z_mb)).numpy()
        value = np.cumsum(EONIA[i, :])[-1] # Value for T-days
        if i % 10 == 0: # Only plot the simulation 2 times
            
            # Show the simulation in the dataset
            simulation = np.reshape(X_hat_scaled, (N, T))
            simulation_cum = np.cumsum(simulation, axis=1)
            
            # Find the nearest neighbour in the dataset
            closest = np.mean(((EONIA - simulation[0, :])**2), axis = 1).argsort()[0]
            equivalent = EONIA[closest, :]
        
            print('Done with ' + str(i) + ' simulations.')
        
        # Train on Synthetic, Test on Real
        if lower:
            VaR_l = value_at_risk(X_hat=X_hat_scaled, percentile=99, upper=False)
            exceedances = exceedances + int(value < VaR_l)
        else:    
            VaR_u = value_at_risk(X_hat=X_hat_scaled, percentile=99, upper=True)
            exceedances = exceedances + int(VaR_u < value)
        
    prob = binom.cdf(np.sum(exceedances), 250, .01)
    
    if prob <= binom.cdf(4, 250, 0.01):
        return 'Green', exceedances
    elif binom.cdf(4, 250, 0.01) <= prob <= binom.cdf(9, 250, 0.01):
        return 'Yellow', exceedances
    else:
        return 'Red', exceedances 
    
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
