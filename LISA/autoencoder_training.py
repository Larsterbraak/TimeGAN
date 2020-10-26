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
epochs = 700
loss_object = tf.keras.losses.MeanSquaredError() 

# 4. Set up the loss and training loop and iterate!
def train(model, opt, original):
  with tf.GradientTape() as tape:
    pred = model(original, True)
    loss_train = loss_object(pred, original)
  gradients = tape.gradient(loss_train, model.trainable_variables)
  opt.apply_gradients(zip(gradients, model.trainable_variables))

#for x in range(1,5):
#    for dropout in np.linspace(0.1, 0.3, 3):    
#        autoencoder = Autoencoder(x, dropout)
#        for epoch in range(epochs):
#            for step, batch_features in enumerate(X_train):
#                train(autoencoder, opt, batch_features)
#            if epoch == epochs-1:
#                total = tf.Variable(0.0)
#                mse = tf.Variable(0.0)
#                for step, batch_features in enumerate(X_train):
#                    total.assign_add(batch_features.shape[0])
#                    mse.assign_add(loss_object(autoencoder(batch_features, False), batch_features) * batch_features.shape[0])
#                print('Finished Autoencoder model with ' + str(x) + ' hidden layers, dropout of: ' + str(dropout) + ' and mean MSE of ' + str(tf.math.divide(mse, total).numpy()) + ' after ' + str(epoch) + ' iterations.')

# 5. Import the Supervisor model 
from models.Supervisor import Supervisor

# 5.1 Initialize the autoencoder and pre-train 10,0000 iterations
autoencoder = Autoencoder(4, 0.1) 

epochs = 1000
for epoch in range(epochs):
    for step, batch_features in enumerate(X_train):
        train(autoencoder, opt, batch_features)
    if epoch == epochs-1:
        total = tf.Variable(0.0)
        mse = tf.Variable(0.0)
        for step, batch_features in enumerate(X_train):
            total.assign_add(batch_features.shape[0])
            mse.assign_add(loss_object(autoencoder(batch_features, False), batch_features) * batch_features.shape[0])
        print('Finished Autoencoder model with 4 latent space, dropout of: 0.1, 2 hidden layers and mean MSE of ' + str(tf.math.divide(mse, total).numpy()) + ' after ' + str(epoch) + ' iterations.')
        
# 5.2 Train the supervsior model for different dropout for 10,000 iterations
def train_supervisor(model, opt, original):
    with tf.GradientTape() as tape:
        pred = model(original)
        loss_train = loss_object(pred[:, 1:, :], original[:, 1:, :])
    gradients = tape.gradient(loss_train, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

#epochs = 500  
#for dropout in np.linspace(0.1, 0.3, 3):
#    supervisor = Supervisor("", [], 4, dropout=dropout)
#    for epoch in range(epochs):
#        for step, batch_features in enumerate(X_train):
#            batch_features = autoencoder.encode(batch_features)
#            train_supervisor(supervisor, opt, batch_features)
#        if epoch == epochs - 1:
#            total = tf.Variable(0.0) 
#            mse = tf.Variable(0.0)
#            for step, batch_features in enumerate(X_train):
#                batch_features = autoencoder.encode(batch_features)
#                total.assign_add(batch_features.shape[0])
#                mse.assign_add(tf.cast(loss_object(supervisor(batch_features)[:, 1:, :],
#                               batch_features[:, 1:, :]), tf.float32) * batch_features.shape[0])
            # print('MSE is ' + str(tf.math.divide(mse, total)))
#            print('Finished Supervisor model with 1 layers in network and 4 latent space and dropout: ' + str(dropout) + ' and mean MSE of ' + str(tf.math.divide(mse, total).numpy()))           