"""
MSc Thesis Quantitative Finance
Title: Interest rate risk simulation using TimeGAN 
       after the EONIA-€STER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: August 19th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Inputs
Data that needs dimension reduction

Outputs
Fully trained autoencoder based on the data
"""

import os
os.chdir('C:/Users/s157148/Documents/GitHub/TimeGAN')

from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import datetime
import time

# 1. Data loading
from data_loading import create_dataset
_, X = create_dataset(name = 'EONIA',
                      normalization = 'min-max',
                      seq_length = 20,
                      training=False,
                      multidimensional=True)

# 2. Import the Embedder and Recovery model and define Autoencoder
from models.Embedder import Embedder
from models.Recovery import Recovery

class Autoencoder(tf.keras.Model):
  def __init__(self, hidden_dim, dropout):
    super(Autoencoder, self).__init__()
    self.encoder = Embedder("", [], hidden_dim, 11, dropout)
    self.decoder = Recovery("", [], hidden_dim, 11, dropout)
  
  def call(self, input_features, training):
    code = self.encoder(input_features, training)
    return self.decoder(code, training)

  def encode(self, input_features):
    return self.encoder(input_features, False)

# 3. Define the optimizers and the hyperparameters
opt = tf.optimizers.Adam(learning_rate=0.01)
epochs = 2
loss_object = tf.keras.losses.MeanSquaredError() 

# 4. Set up the loss and training loop and iterate!
def train_model(model, opt, original):
    with tf.GradientTape() as tape:
        pred = model(original, True)
        loss_train = loss_object(pred, original)
    gradients = tape.gradient(loss_train, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

now = time.time()

for x in range(1,2):
    for dropout in np.linspace(0.1, 0.1, 1):
        results = []
        
        kf = KFold(n_splits=10)
        for train, test in kf.split(X):
            X_train = X[train, :, :]
            X_test = X[test, :, :]
            
            X_train = tf.data.Dataset.from_tensor_slices(tf.cast(X_train, tf.float32)).batch(32)
            X_test = tf.data.Dataset.from_tensor_slices(tf.cast(X_test, tf.float32)).batch(32)
            autoencoder = Autoencoder(x, dropout)
            
            for epoch in range(epochs):
                for step, batch_features in enumerate(X_train):
                    train_model(autoencoder, opt, batch_features)
                if epoch == epochs -1:
                    total = tf.Variable(0.0)
                    mse = tf.Variable(0.0)
                    for step, batch_features in enumerate(X_test):
                        total.assign_add(batch_features.shape[0])
                        mse.assign_add(loss_object(autoencoder(batch_features, False), batch_features) * batch_features.shape[0])
                    
                    results.append(tf.math.divide(mse, total).numpy())
                print(epoch)
        print('Finished Autoencoder model with ' + str(x) + ' hidden layers, dropout of: ' + str(dropout) + ' and mean MSE of ' + str(np.round(np.mean(results),6)) + ' and std MSE of ' + str(np.round(np.std(results), 6)))

elapsed = time.time() - now
print('One hyperparameter setting took' + str(np.round(elapsed,1)) + ' seconds')

# 5. Import the Supervisor model 
from models.Supervisor import Supervisor

# 5.1 Initialize the autoencoder and pre-train 10,0000 iterations
autoencoder = Autoencoder(3, 0.1) 
epochs = 300

X_train = tf.data.Dataset.from_tensor_slices(tf.cast(X, tf.float32)).batch(32)
for epoch in range(epochs):
    for step, batch_features in enumerate(X_train):
        train_model(autoencoder, opt, batch_features)
    print(epoch)
        
# 5.1.1 Visualize the embeddings produced by the autoencoder network
df = pd.read_csv("data/Master_EONIA.csv", sep=";")   
df_process = preprocessing.MinMaxScaler().fit_transform(df.iloc[:, 1:]) 
latent_space = autoencoder.encode(df_process[3548:3568,:].reshape((1,20,11,))).numpy()

# More efficient and elegant pyplot
plt.style.use(['science', 'no-latex'])
dates = np.ravel(df.Date[3548:3568].values).astype(str)
dates = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
               for d in dates]

from matplotlib import rcParams
rcParams['axes.titlepad']=20
rcParams['ytick.labelsize']=24
rcParams['xtick.labelsize']=16

plt.figure(figsize=(12,8), dpi=500)
plt.plot_date(dates, df.EONIA.iloc[3548:3568].values, 'b-', color = '#0C5DA5') 
plt.plot_date(dates, latent_space.reshape(20,5)[:, 0]/14 -0.38, 'b-', color = '#00B945')
plt.plot_date(dates, latent_space.reshape(20,5)[:, 1]/14 -0.38, 'b-', color = '#FF9500')
plt.plot_date(dates, latent_space.reshape(20,5)[:, 2]/14 -0.38, 'b-', color = '#FF2C00')
plt.plot_date(dates, latent_space.reshape(20,5)[:, 3]/14 -0.38, 'b-', color = '#845B97')
plt.plot_date(dates, latent_space.reshape(20,5)[:, 4]/14 -0.38, 'b-', color = '#474747')
ax = plt.gca()
ax.set_xlim(datetime.date(2017, 10, 6), datetime.date(2017, 11, 2))

def to_transactions(x):
    return (x + 0.38)*14
        
def to_volume(x):
    return x/14 -0.38

ax.set_ylim((-.38, -.295))
secaxy = ax.secondary_yaxis('right', functions = (to_transactions, to_volume))
secaxy.set_ylabel('Latent variable weight', fontsize=26, fontweight='roman', labelpad=20)
plt.legend(('EONIA', '$\mathcal{H}_1$', '$\mathcal{H}_2$',
            '$\mathcal{H}_3$', '$\mathcal{H}_4$', '$\mathcal{H}_5$'), 
           fontsize = 'xx-large')
plt.ylabel('Short rate [%]', fontsize=26, fontweight='roman', labelpad=20)
plt.xlabel('Time ', fontsize=26, fontweight='roman', labelpad=20)
plt.title(r'EONIA and latent variables over time', fontsize=30, 
          fontweight='roman')
#plt.savefig('EONIA_LATENT.png')
plt.show()

# Increase the contrast of the picture
from PIL import Image, ImageEnhance
im1 = Image.open('EONIA_latent_variables_5_DIM.png')
enhancer = ImageEnhance.Brightness(im1)

enhancer2 = ImageEnhance.Contrast(im1)
factor = 1.5
im_output = enhancer2.enhance(factor)
im_output.save('EONIA_latent_variables_5_DIM_contrast.png')

# 5.2 Train the supervsior model for different dropout for 10,000 iterations
def train_supervisor(model, opt, original):
    with tf.GradientTape() as tape:
        pred = model(original)
        loss_train = loss_object(pred[:, 1:, :], original[:, 1:, :])
    gradients = tape.gradient(loss_train, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 100
for dropout in np.linspace(0.1, 0.3, 3):
    supervisor = Supervisor("", [], 3, dropout=dropout)
    for epoch in range(epochs):
        for step, batch_features in enumerate(X_train):
            batch_features = autoencoder.encode(batch_features)
            train_supervisor(supervisor, opt, batch_features)
        print('Current epoch:' + str(epoch+1))
        if epoch == epochs -1:
            total = tf.Variable(0.0) 
            mse = tf.Variable(0.0)
            for step, batch_features in enumerate(X_train):
                batch_features = autoencoder.encode(batch_features)
                total.assign_add(batch_features.shape[0])
                mse.assign_add(loss_object(supervisor(batch_features, False)[:, 1:, :],
                                   batch_features[:, 1:, :]) * batch_features.shape[0])
            # print('MSE is ' + str(tf.math.divide(mse, total)))
            print('Finished Supervisor model with dropout: ' + str(dropout) + ' and mean MSE of ' + str(tf.math.divide(mse, total).numpy()))

# Implement Bayesian optimization for Supervisor network
# import sklearn.gaussian_process as gp
        
# def bayesian_optimization(n_iters, sample_loss, xp, yp):
#   """

#   Arguments:
#   ----------
#     n_iters: int.
#       Number of iterations to run the algorithm for.
#     sample_loss: function.
#       Loss function that takes an array of parameters.
#     xp: array-like, shape = [n_samples, n_params].
#       Array of previously evaluated hyperparameters.
#     yp: array-like, shape = [n_samples, 1].
#       Array of values of `sample_loss` for the hyperparameters
#       in `xp`.
#   """

#   # Define the GP
#   kernel = gp.kernels.Matern()
#   model = gp.GaussianProcessRegressor(kernel=kernel,
#                                       alpha=1e-4,
#                                       n_restarts_optimizer=10,
#                                       normalize_y=True)
#   for i in range(n_iters):
#     # Update our belief of the loss function
#     model.fit(xp, yp)

#     # sample_next_hyperparameter is a method that computes the arg
#     # max of the acquisition function
#     next_sample = sample_next_hyperparameter(model, yp)

#     # Evaluate the loss for the new hyperparameters
#     next_loss = sample_loss(next_sample)

#     # Update xp and yp    