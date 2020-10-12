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
VaR(99%) based on Kalman Filter for 20 day prediction
VaR(99%) based on Linear Regression estimation of 1-factor Vasicek for 20 day prediction
"""

import os
os.chdir('C:/Users/s157148/Documents/GitHub/TimeGAN')

import pandas as pd
import numpy as np
from sklearn import preprocessing
from metrics import load_models, create_plot_nn
from training import RandomGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from descriptions_tensorboard import descr_images
import datetime

# 1. Import the EONIA data
data = pd.read_csv("data/EONIA.csv", sep=";")
dates_EONIA = np.ravel(data.Date[:505][::-1].values).astype(str)
dates_EONIA = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
               for d in dates_EONIA]
data = np.array(data.iloc[:505,2])[::-1] # Filter for only EONIA

# Import the data and apply the transformation
# We import the data until 6-10-2017
df = pd.read_csv("data/Master_EONIA.csv", sep=";")
df = df.iloc[:, 1:] # Remove the Date variable from the dataset
df.EONIA[1:] = np.diff(df.EONIA)
df = df.iloc[1:, :]
scaler = preprocessing.MinMaxScaler().fit(df)

# Define the settings
hparams = []
hidden_dim = 4
epochs = np.linspace(150, 9150, 61).astype(int)
epochs = np.delete(epochs, [12])

for epoch in epochs:
    load_epochs = epoch

    # Import the pre-trained models
    embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(load_epochs, hparams, hidden_dim)
    
    # # Do the VaR(99.5%) calculations
    upper = []
    lower = []
    for j in range(504):
        Z_mb = RandomGenerator(10000, [1, hidden_dim])
        samples = recovery_model(generator_model(Z_mb)).numpy()
        reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                         samples.shape[2]))
        
        scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
        simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                                     samples.shape[1], 
                                                     samples.shape[2])))
        
        results = np.sum(simulations[:,:,8], axis=1)
        
        results.sort()
        upper.append(results[9900])
        lower.append(results[100])
        print(j)
    
    differences = []
    for i in range(504):
        differences.append(data[i+1] - data[i])

    upper_exceed = []    
    lower_exceed = []    
    for i in range(504):
        upper_exceed.append(data[i+1] > (data[i] + upper[i]))
        lower_exceed.append(data[i+1] < (data[i] + lower[i]))
        
    upper_exceedances = np.sum(upper_exceed)
    lower_exceedances = np.sum(lower_exceed)  

    # Do the nearest neighbours calculations
    Z_mb = RandomGenerator(504, [1, hidden_dim])
    simulated = recovery_model(generator_model(Z_mb)).numpy()

    # Check the number of distinct neighbours
    neighbours = np.zeros(shape=(504,))
    for i in range(504):
        number=0
        simulation = scaler.inverse_transform(simulated[i,:,:])[:,8]
        best = np.mean((simulation - differences[number])**2)
        for j in range(1, 504):
            if np.mean((simulation -  differences[j])**2) < best:
                best = np.mean((simulation -  differences[j])**2)
                number = j
            neighbours[i] = number
    
    uniques = np.unique(neighbours)
    diversity = int(len(uniques))
    
    # More efficient and elegant pyplot
    plt.style.use(['science', 'no-latex'])
    
    from matplotlib import rcParams
    rcParams['axes.titlepad']=10
    rcParams['ytick.labelsize']=20
    rcParams['xtick.labelsize']=16
    
    textstr = '\n'.join((
        r'Upper exceedances$=%.0f$' % (upper_exceedances, ),
        r'Lower exceedances$=%.0f$' % (lower_exceedances, ),
        r'Diversity$=%.0f$' % (diversity)))
    
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.plot_date(dates_EONIA[1:], upper, 'b-', color = '#0C5DA5')
    ax.plot_date(dates_EONIA[1:], lower, 'b-', color = '#00B945')
    ax.plot_date(dates_EONIA[1:], differences, 'b-',  color = '#FF9500')
    plt.title('1-day VaR(99%) for EONIA after ' + str(epoch) + ' epochs',
              fontsize=24, fontweight='roman')
    plt.xlabel(r'Time $t$', fontsize=20, fontweight='roman', labelpad=10)
    plt.ylabel(r'$\Delta r_t$', fontsize=20, fontweight='roman', labelpad=10)
    plt.legend(['upper 99%-VaR', 'lower 99%-VaR', 'EONIA'], loc='lower left',
               prop={'size': 24})
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=24,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    plt.show()

# Make a GIF of the VaR predictions during the training iterations
from PIL import Image

img_300 = Image.open('logs/PLM_FM/300.png')
img_450 = Image.open('logs/PLM_FM/450.png')
img_600 = Image.open('logs/PLM_FM/600.png')
img_750 = Image.open('logs/PLM_FM/750.png')
img_900 = Image.open('logs/PLM_FM/900.png')
img_300 = Image.open('logs/PLM_FM/300.png')
img_450 = Image.open('logs/PLM_FM/450.png')
img_600 = Image.open('logs/PLM_FM/600.png')
img_750 = Image.open('logs/PLM_FM/750.png')
img_900 = Image.open('logs/PLM_FM/900.png')

images = [img_300, img_450, img_600, img_750, img_900]
# Append all images to this array

images[0].save('PLS+FM.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=2000, loop=10)



# Insert the image for the Tensorboard because the server does not allow
# us to make pictures

def create_figure():
    figure = plt.figure(dpi=600,figsize=(25,4))
    plt.title('Nearest neighbour in the EONIA data')
    for i in range(20):
        
        # N simulations for TimeGAN calibrated on EONIA
        Z_mb = RandomGenerator(1, [20, hidden_dim])
        X_hat_scaled = recovery_model(generator_model(Z_mb)).numpy()
        simulation = scaler.inverse_transform(X_hat_scaled[0,:,:])[:,8]
        
        # Find the nearest neighbour in the dataset
        closest = np.mean(((np.transpose(real) - simulation)**2), axis = 1).argsort()[0]
        equivalent = real[:,closest]
        
        # Start next subplot.
        if i < 4:
            ax = plt.subplot(1, 4, i + 1)#, title=str('Nearest Neighbour '+ str(i+1) ) )
            create_plot_nn(equivalent, simulation, 20, ax)
        plt.grid(False)
    plt.tight_layout()
    return figure

figure = create_figure()
figure.canvas.draw()
w, h = figure.canvas.get_width_height()
img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
img_1 = img.reshape((1, h, w, 3))  

figure = create_figure()
figure.canvas.draw()
w, h = figure.canvas.get_width_height()
img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
img_2 = img.reshape((1, h, w, 3))  

figure = create_figure()
figure.canvas.draw()
w, h = figure.canvas.get_width_height()
img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
img_3 = img.reshape((1, h, w, 3))  

figure = create_figure()
figure.canvas.draw()
w, h = figure.canvas.get_width_height()
img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
img_4 = img.reshape((1, h, w, 3))  

img = np.array([img_1, img_2, img_3, img_4]).reshape((4,h,w,3)) 

# Crop the images
img = img[:, :, :1000, :]

#log_dir = 'logs/Normal_training_final/Normal_1_disc_step_0_05_gen_0_0008_disc_20201006-125054'
#log_dir = 'logs/Positive_label_smoothing_final/PLM,lrG-0.02,lrD-0.0008,minmax,20201006-164312'
#log_dir = 'logs/PLM_FM/ALL,lrG-0.02,lrD-0.0008,minmax,20201006-201922'
log_dir = 'logs/WGAN_GP_final/WGANGP5Dsteps,lrG-0.0005,lrD-0.0005,minmax,gamma0.01,oldnormcalc,withFM,1e-4forsqrt,20201007-203612'

summary_writer_train = tf.summary.create_file_writer(log_dir + '/image')
with summary_writer_train.as_default():
    tensor = tf.constant(img)
    tf.summary.image(str("Nearest neighbours after " + str(load_epochs) + " epochs"),
                     tensor, step=load_epochs, max_outputs=4, 
                     description = str(descr_images()))