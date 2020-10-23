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
from kalman_filter_vasicek import VaR as VaR_vasicek
from variance_covariance import VaR as VaR_var_covar
from matplotlib import rcParams 
from stylized_facts import descriptives

# 1. Import the EONIA data
data = pd.read_csv("data/EONIA.csv", sep=";")
dates_EONIA = np.ravel(data.Date[:525][::-1].values).astype(str)
dates_EONIA = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
               for d in dates_EONIA]
data = np.array(data.iloc[:525,2])[::-1] # Filter for only EONIA

# Import the data and apply the transformation
# We import the data until 6-10-2017
df = pd.read_csv("data/Master_EONIA.csv", sep=";")
dates_t_var = df.iloc[3798:, 0]
dates_t_var = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
               for d in dates_t_var]
df = df.iloc[:, 1:] # Remove the Date variable from the dataset
df.EONIA[1:] = np.diff(df.EONIA)
df = df.iloc[1:, :]
scaler = preprocessing.MinMaxScaler().fit(df)

# Define the settings
hparams = []
hidden_dim = 4
epochs = [2700]
#epochs = np.linspace(150, 9600, 64).astype(int)
#hidden_dim = 3
#epochs = np.linspace(50, 150, 3).astype(int)

for epoch in epochs:
    load_epochs = epoch

    # Import the pre-trained models
    embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(load_epochs, hparams, hidden_dim)
    
    # # Do the VaR(99.5%) calculations
    upper = []
    lower = []
    for j in range(250):
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
    for i in range(251, 501):
        differences.append(data[i+1] - data[i])

    upper_exceed = []    
    lower_exceed = []    
    for i in range(251, 501):
        upper_exceed.append(data[i+1] > (data[i] + upper[i-251]))
        lower_exceed.append(data[i+1] < (data[i] + lower[i-251]))
        
    upper_exceedances = np.sum(upper_exceed)
    lower_exceedances = np.sum(lower_exceed)  

    # Do the nearest neighbours calculations
    Z_mb = RandomGenerator(250, [1, hidden_dim])
    simulated = recovery_model(generator_model(Z_mb)).numpy()

    # Check the number of distinct neighbours
    neighbours = np.zeros(shape=(250,))
    for i in range(250):
        number=0
        simulation = scaler.inverse_transform(simulated[i,:,:])[:,8]
        best = np.mean((simulation - differences[number])**2)
        for j in range(250):
            if np.mean((simulation -  differences[j])**2) < best:
                best = np.mean((simulation -  differences[j])**2)
                number = j
        neighbours[i] = number
    
    uniques = np.unique(neighbours)
    diversity = int(len(uniques))
    print('The exceedances, upper: ' + str(upper_exceedances) +
          ' lower: ' + str(lower_exceedances))
    print('The diversity is: ' + str(diversity))
    
    # More efficient and elegant pyplot
    plt.style.use(['science', 'no-latex'])
    rcParams['axes.titlepad']=10
    rcParams['ytick.labelsize']=20
    rcParams['xtick.labelsize']=16
    
    textstr = '\n'.join((
        r'Upper exceedances$=%.0f$' % (upper_exceedances, ),
        r'Lower exceedances$=%.0f$' % (lower_exceedances, ),
        r'Diversity$=%.0f$' % (diversity)))
    
    fig, ax = plt.subplots(1, figsize=(10,10), dpi=300)
    ax.plot_date(dates_EONIA[20:], upper, 'b-', color = '#0C5DA5')
    ax.plot_date(dates_EONIA[20:], lower, 'b-', color = '#00B945')
    ax.plot_date(dates_EONIA[20:], differences, 'b-',  color = '#FF9500')
    plt.title('20-day VaR(99%) for EONIA after ' + str(epoch) + ' epochs',
              fontsize=24, fontweight='roman')
    plt.xlabel(r'Time $t$', fontsize=20, fontweight='roman', labelpad=10)
    plt.ylabel(r'$\Delta r_t$', fontsize=20, fontweight='roman', labelpad=10)
    plt.legend(['upper 99%-VaR', 'lower 99%-VaR', 'EONIA'], loc='lower left',
               prop={'size': 24})
    
    # Add a line to distinguish between the train and test set
    ax.axvline(x=dates_EONIA[250], ymin=0, ymax=1, ls='--', color='red')
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=24,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    plt.show()

for epoch in epochs:
    for date in range(250):
        
        # Import the pre-trained models
        embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(epoch, hparams, hidden_dim)
        
        # Make a VaR plot for multiple T
        upper = []
        lower = []
            
        for time in range(1, 21):
            Z_mb = RandomGenerator(10000, [20, hidden_dim])
            samples = recovery_model(generator_model(Z_mb)).numpy()
            reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                             samples.shape[2]))
            
            scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
            simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                                         samples.shape[1], 
                                                         samples.shape[2])))
            
            results = np.sum(simulations[:,:time,8], axis=1)
            
            results.sort()
            upper.append(results[9900])
            lower.append(results[100])
            print(time)
            
        # Compute the Kalman filtered Vasicek models
        upper_VaR_vasicek = [VaR_vasicek(r_0=data[250+date], data=data[date:250+date], 
                                 time=x, upward=True) for x in range(1,21)] - data[250+date]
        lower_VaR_vasicek = [VaR_vasicek(r_0=data[250+date], data=data[date:250+date],
                                 time=x, upward=False) for x in range(1,21)] - data[250+date]
    
        # Compute the Variance-Covariance method models
        # We have to scale the variance in order to get good performance
        current_situation = np.array(df.drop('EONIA', axis=1).iloc[3527+date,:])
        upper_VaR_var_covar = [VaR_var_covar(data[250+date], current_situation, df.iloc[3277+date:3527+date, :], x, 
                                             0.99, True) for x in range(1,21)] - data[250+date]    
        
        lower_VaR_var_covar =[VaR_var_covar(data[250+date], current_situation, df.iloc[3277+date:3527+date, :], x, 
                                             0.99, False) for x in range(1,21)] - data[250+date]    
        
        # More efficient and elegant pyplot
        plt.style.use(['science', 'no-latex'])
        rcParams['axes.titlepad']=7
        rcParams['ytick.labelsize']=10
        rcParams['xtick.labelsize']=10
        
        plt.figure(dpi=400)
        plt.title('EONIA T-VaR(99%) at ' + str(dates_EONIA[250+date]))
        plt.xlabel('T', fontsize=12, fontweight='roman', labelpad=4)
        plt.ylabel(r'$\Delta r_t$', fontsize=12, fontweight='roman', labelpad=4)
        plt.ylim(-0.065, 0.115)
        plt.plot(range(1,21), upper, color = '#0C5DA5', label='Upper VaR(99%) TimeGAN PLS + FM')
        plt.plot(range(1,21), lower, color = '#00B945', label='Lower VaR(99%) TimeGAN PLS+ FM')
        plt.plot(range(1,21), upper_VaR_vasicek, color='#FF9500', label='Upper VaR(99%) Vasicek')
        plt.plot(range(1,21), lower_VaR_vasicek, color='#FF2C00', label='Lower VaR(99%) Vasicek')
        plt.plot(range(1,21), upper_VaR_var_covar, color='#845B97', label='Upper VaR(99%) Var-Covar')
        plt.plot(range(1,21), lower_VaR_var_covar, color='#474747', label='Lower VaR(99%) Var-Covar')
        plt.hlines(y=0, xmin=1, xmax=20, color='black', linestyle='dashed')
        plt.legend(loc='upper left', fontsize='xx-small')
        plt.show()

# Check what the influence is of the latent space
epoch = 8250

# Import the pre-trained models
embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(epoch, hparams, hidden_dim)    

# Fix the random generator at a specific point
Z_mb = np.zeros(shape=(1000, 20, 4))

changes = np.linspace(0, 1, 5)

fig, axs = plt.subplots(5, 4, figsize=(25,10), sharey=True)
rcParams['ytick.labelsize']=8
rcParams['xtick.labelsize']=8

for i in range(5):
    latent_space = generator_model(Z_mb, False).numpy()
    
    # Adjust the first dimension of the latent space
    latent_space[:, :, 0] = latent_space[:, :, 0] + changes[i]
    
    samples = recovery_model(latent_space).numpy()
    reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                     samples.shape[2]))
            
    scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
    simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                                 samples.shape[1], 
                                                 samples.shape[2])))
    
    axs[i, 0].plot(range(1,21), simulations[:,:,8].transpose(), color= '#0C5DA5')
    axs[i, 0].set(xlabel='T', ylabel=r'$\Delta r_t$', ylim=(-0.15,0.20),
                  title = r'20-day EONIA')
    axs[i, 0].legend([r'$\mathcal{H}_1 =$ ' +str(np.round(changes[i],2))])
    
for i in range(5):
    latent_space = generator_model(Z_mb, False).numpy()
    
    # Adjust the first dimension of the latent space
    latent_space[:, :, 1] = latent_space[:, :, 1] + changes[i]
    
    samples = recovery_model(latent_space).numpy()
    reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                     samples.shape[2]))
            
    scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
    simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                                 samples.shape[1], 
                                                 samples.shape[2])))

    axs[i, 1].plot(range(1,21), simulations[:,:,8].transpose(), color='#00B945')
    axs[i, 1].set(xlabel='T', ylabel=r'$\Delta r_t$', ylim=(-0.15,0.20),
                  title = r'20-day EONIA') 
    axs[i, 1].legend([r'$\mathcal{H}_2 =$ ' +str(np.round(changes[i],2))])

for i in range(5):
    latent_space = generator_model(Z_mb, False).numpy()
    
    # Adjust the first dimension of the latent space
    latent_space[:, :, 2] = latent_space[:, :, 2] + changes[i]
    
    samples = recovery_model(latent_space).numpy()
    reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                     samples.shape[2]))
            
    scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
    simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                                 samples.shape[1], 
                                                 samples.shape[2])))
    
    axs[i, 2].plot(range(1,21), simulations[:,:,8].transpose(), color='#FF9500')
    axs[i, 2].set(xlabel='T', ylabel=r'$\Delta r_t$', ylim=(-0.15,0.20),
                  title = r'20-day EONIA')  
    axs[i, 2].legend([r'$\mathcal{H}_3 =$ ' +str(np.round(changes[i],2))])

for i in range(5):
    latent_space = generator_model(Z_mb, False).numpy()
    
    # Adjust the first dimension of the latent space
    latent_space[:, :, 3] = latent_space[:, :, 3] + changes[i]
    
    samples = recovery_model(latent_space).numpy()
    reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                     samples.shape[2]))
            
    scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
    simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                                 samples.shape[1], 
                                                 samples.shape[2])))
    
    axs[i, 3].plot(range(1,21), simulations[:,:,8].transpose(), color='#FF2C00')
    axs[i, 3].set(xlabel='T', ylabel=r'$\Delta r_t$', ylim=(-0.15,0.20),
                  title = r'20-day EONIA')
    axs[i, 3].legend([r'$\mathcal{H}_4 =$ ' +str(np.round(changes[i],2))])

fig.tight_layout()

# Check which stylized facts change for latent space deviations
epoch = 8250

# Import the pre-trained models
embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(epoch, hparams, hidden_dim)

# # Define the correlations matrix for the latent variables
# correlations_latent = np.zeros(shape=(4, 7))

# # Fix the random generator at a specific point
# Z_mb = np.zeros(shape=(10000, 20, 4))

# # Define the changes in the latent space
# changes = np.linspace(0.1, 0.9, 5)

# for latent_dim in range(4):
#     result_latent = np.zeros(shape=(5, 7))
#     for i in range(5):    
#         latent_space = generator_model(Z_mb, False).numpy()
            
#         # Adjust the first dimension of the latent space
#         latent_space[:, :, 1] = latent_space[:, :, 1] - changes[i]
            
#         samples = recovery_model(latent_space).numpy()
#         reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
#                                          samples.shape[2]))
                    
#         scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
#         simulations = scaled_reshaped_data.reshape(((samples.shape[0],
#                                                      samples.shape[1], 
#                                                      samples.shape[2])))
        
#         sf_latent_space = np.zeros(shape=(10000, 7))
        
#         for j in range(10000):
#             sf_latent_space[j, :] = descriptives(simulations[j,:,8])
        
#         result_latent[i, :] = np.mean(sf_latent_space, axis=0)
#         print(i)

#     for l in range(7):    
#         correlations_latent[latent_dim, l] = np.corrcoef(np.linspace(0,1,5), 
#                                                          result_latent[:,l])[0,1]    

# Make a t-SNE visualizations of the potential hidden space
# and the generated hidden space
simulations = 5000
hidden_dim = 4

potentials = np.random.uniform(0., 1, [simulations, hidden_dim])

Z_mb = RandomGenerator(int(simulations/20), [20, hidden_dim])
samples = np.reshape(generator_model(Z_mb).numpy(), newshape=(simulations, 
                                                              hidden_dim))

# Reshape back to [2*counter, 20*hidden_dim]
x = np.concatenate((potentials, samples), axis=0)
y = np.append(['potentials' for x in range(simulations)], 
              ['samples' for x in range(simulations)])

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

tsne = TSNE(n_components=2, verbose=1, perplexity=40.0, n_iter=300, n_jobs=-1)
tsne_results = tsne.fit_transform(X=x)

feat_cols = ['point'+str(i) for i in range(x.shape[1])]
df = pd.DataFrame(x, columns=feat_cols)
df['y'] = y
df['tsne-one'] = tsne_results[:,0]
df['tsne-two'] = tsne_results[:,1]

plt.figure(dpi=600, figsize=(16,10))
ax=sns.scatterplot(
        x='tsne-one', y='tsne-two',
        hue='y',
        palette=['dodgerblue', 'red'],
        data=df,
        legend="full",
        alpha=1,
        s=30
    )
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])
plt.setp(ax.get_legend().get_texts(), fontsize='26')
plt.axis('off')
plt.show()

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