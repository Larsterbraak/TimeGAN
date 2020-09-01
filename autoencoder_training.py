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

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import datetime

# 1. Data loading
from data_loading import create_dataset
X_train, X_test = create_dataset(name = 'EONIA',
                                 normalization = 'outliers',
                                 seq_length = 20,
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
def train(model, opt, original):
  with tf.GradientTape() as tape:
    pred = model(original, True)
    loss_train = loss_object(pred, original)
  gradients = tape.gradient(loss_train, model.trainable_variables)
  opt.apply_gradients(zip(gradients, model.trainable_variables))

for x in range(1,5):
    for dropout in np.linspace(0.1, 0.3, 3):    
        autoencoder = Autoencoder(x, dropout)
        for epoch in range(epochs):
            for step, batch_features in enumerate(X_train):
                train(autoencoder, opt, batch_features)
            if epoch == epochs -1:
                total = tf.Variable(0.0)
                mse = tf.Variable(0.0)
                for step, batch_features in enumerate(X_train):
                    total.assign_add(batch_features.shape[0])
                    mse.assign_add(loss_object(autoencoder(batch_features, False), batch_features) * batch_features.shape[0])
                #print('MSE is ' + str(tf.math.divide(mse, total)))
                print('Finished Autoencoder model with ' + str(x) + ' hidden layers, dropout of: ' + str(dropout) + ' and mean MSE of ' + str(tf.math.divide(mse, total).numpy()))
        
# 5. Import the Supervisor model 
from models.Supervisor import Supervisor

# 5.1 Initialize the autoencoder and pre-train 10,0000 iterations
autoencoder = Autoencoder(4, 0.1) 
epochs = 100
for epoch in range(epochs):
    for step, batch_features in enumerate(X_train):
        train(autoencoder, opt, batch_features)
    print(epoch)
        
# 5.1.1 Visualize the embeddings produced by the autoencoder network
df = pd.read_csv("data/Master_EONIA.csv", sep=";")   
df_process = preprocessing.RobustScaler().fit_transform(df.iloc[:, 1:]) 
latent_space = autoencoder.encode(df_process[3548:3568,:].reshape((1,20,11,))).numpy()

# More efficient and elegant pyplot
plt.style.use(['science', 'no-latex'])

dates = np.ravel(df.Date[3548:3568].values).astype(str)
dates = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
               for d in dates]

plt.figure(figsize=(12,8))
plt.plot_date(dates, df.EONIA.iloc[3548:3568].values, 'b-', color = '#0C5DA5') 
plt.plot_date(dates, latent_space.reshape(20,4)[:, 0]/10 -0.42, 'b-', color = '#00B945')
plt.plot_date(dates, latent_space.reshape(20,4)[:, 1]/10 -0.42, 'b-', color = '#FF9500')
plt.plot_date(dates, latent_space.reshape(20,4)[:, 2]/10 -0.42, 'b-', color = '#FF2C00')
plt.plot_date(dates, latent_space.reshape(20,4)[:, 3]/10 -0.42, 'b-', color = '#845B97')
ax = plt.gca()
ax.set_xlim(datetime.date(2017, 10, 5), datetime.date(2017, 11, 3))

def to_transactions(x):
    return (x + 0.42)*10
        
def to_volume(x):
    return x/10 -0.42

ax.set_ylim((-.385, -.345))
secaxy = ax.secondary_yaxis('right', functions = (to_transactions, to_volume))
secaxy.set_ylabel(r'Latent variables')
plt.legend(('EONIA', '$\mathcal{H}_1$', '$\mathcal{H}_2$',
            '$\mathcal{H}_3$', '$\mathcal{H}_4$'), fontsize = 'xx-large')
plt.ylabel(r'Short rate $r_t$ [%]')
plt.xlabel(r'Time $t$')
plt.title(r'EONIA and latent variables over time')
#plt.savefig('EONIA_LATENT.png')
plt.show()

# 5.2 Train the supervsior model for different dropout for 10,000 iterations
def train_supervisor(model, opt, original):
    with tf.GradientTape() as tape:
        pred = model(original)
        loss_train = loss_object(pred[:, 1:, :], original[:, 1:, :])
    gradients = tape.gradient(loss_train, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 3    
for dropout in np.linspace(0.1, 0.3, 3):
    supervisor = Supervisor("", [], 2, dropout=dropout)
    for epoch in range(epochs):
        for step, batch_features in enumerate(X_train):
            batch_features = autoencoder.encode(batch_features)
            train_supervisor(supervisor, opt, batch_features)
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
    