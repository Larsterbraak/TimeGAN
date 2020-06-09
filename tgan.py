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

import numpy as np
import datetime
import tensorflow as tf
import os

# Change to the needed working directory
os.chdir('C://Users/s157148/Documents/Github/TimeGAN')

# 3. Train TimeGAN model
from models.Embedder import Embedder
from models.Recovery import Recovery
from models.Generator import Generator
from models.Supervisor import Supervisor
from models.Discriminator import Discriminator
from training import add_hist, RandomGenerator

from descriptions_tensorboard import (descr_auto_loss, descr_auto_grads_embedder,
                                      descr_auto_grads_recovery, descr_supervisor_loss,
                                      descr_auto_grads_supervisor, descr_auto_loss_joint_auto,
                                      descr_auto_loss_joint_supervisor, descr_supervisor_loss_joint,
                                      descr_generator_loss_joint, descr_discriminator_loss_joint,
                                      descr_accuracy_joint, descr_joint_grad_discriminator,
                                      descr_joint_grad_generator, descr_images)

from metrics import load_models, image_grid

# Special function
from tensorflow.keras import backend as K

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis=-1)

def gradient_penalty(self, f, real, fake):
        alpha = np.random.uniform([batch_size, 1, 1, 1], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

def run(parameters, hparams, X_train, X_test, 
        load=False, load_epochs=0, load_log_dir=""):
    
    # Network Parameters
    hidden_dim   = parameters['hidden_dim']
    num_layers   = parameters['num_layers']  # Still have to implement
    iterations   = parameters['iterations']  # Test run to check for overfitting
    batch_size   = parameters['batch_size']  # Currently locked at 25
    module_name  = parameters['module_name'] # 'lstm' or 'GRU'' --> Still have to implement this
    z_dim        = parameters['z_dim']       
    lambda_val   = 1
    eta          = 1
    kappa = 1
    
    if load: # Write to already defined log directory?
        log_dir = load_log_dir
    else: # Or create new log directory?
        # Define the TensorBoard such that we can visualize the results
        log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    summary_writer_train = tf.summary.create_file_writer(log_dir + '/train')
    summary_writer_test = tf.summary.create_file_writer(log_dir + '/test')
    summary_writer_bottom = tf.summary.create_file_writer(log_dir + '/bottom')
    summary_writer_top = tf.summary.create_file_writer(log_dir + '/top')
    summary_writer_real_data = tf.summary.create_file_writer(log_dir + '/real_data')
    summary_writer_fake_data = tf.summary.create_file_writer(log_dir + '/fake_data')
    summary_writer_lower_bound = tf.summary.create_file_writer(log_dir + '/lower_bound')
    
    if load:
        embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(load_epochs)
    else:
        # Create an instance of all neural networks models (All LSTM)
        embedder_model = Embedder('logs/embedder', hparams, dimensionality = 1)
        recovery_model = Recovery('logs/recovery', hparams, dimensionality = 1) # If used for EONIA rate only
        generator_model = Generator('logs/generator', hparams)
        supervisor_model = Supervisor('logs/supervisor', hparams)
        discriminator_model = Discriminator('logs/TimeGAN', hparams)
        
    # Because of a technicality
    load_epochs = load_epochs + 50
        
    r_loss_train = tf.keras.metrics.Mean(name='r_loss_train') # Step 1 metrics 
    r_loss_test = tf.keras.metrics.Mean(name='r_loss_test')
    
    grad_embedder_ll = tf.keras.metrics.Mean(name='e_grad_lower_layer') # Step 1 gradient
    grad_embedder_ul = tf.keras.metrics.Mean(name='e_grad_upper_layer')
    grad_recovery_ll = tf.keras.metrics.Mean(name='r_grad_lower_layer')
    grad_recovery_ul = tf.keras.metrics.Mean(name='r_grad_upper_layer')
    
    g_loss_s_train = tf.keras.metrics.Mean(name='g_loss_s_train') # Step 2 metrics
    g_loss_s_test = tf.keras.metrics.Mean(name='g_loss_s_test')
    
    grad_supervisor_ll = tf.keras.metrics.Mean(name='s_grad_lower_layer') # Step 2 gradients
    grad_supervisor_ul = tf.keras.metrics.Mean(name='s_grad_upper_layer')
    
    e_loss_T0 = tf.keras.metrics.Mean(name='e_loss_T0') # Step 3 metrics (train)
    g_loss_s_embedder = tf.keras.metrics.Mean(name='g_loss_s_embedder')
    g_loss_s = tf.keras.metrics.Mean(name='g_loss_s')
    d_loss = tf.keras.metrics.Mean(name='d_loss')
    g_loss_u_e = tf.keras.metrics.Mean(name='g_loss_u_e')
    
    e_loss_T0_test = tf.keras.metric.Mean(name='e_loss_T0_test') # Step 3 metrics (test)
    g_loss_s_embedder_test = tf.keras.metric.Mean(name='e_loss_T0_test')

    grad_discriminator_ll = tf.keras.metrics.Mean(name='d_grad_lower_layer') # Step 3 gradients
    grad_discriminator_ul = tf.keras.metrics.Mean(name='d_grad_upper_layer')
    grad_generator_ll = tf.keras.metrics.Mean(name='g_grad_lower_layer')
    grad_generator_ul = tf.keras.metrics.Mean(name='g_grad_upper_layer')
    
    loss_object_accuracy = tf.keras.metrics.Accuracy() # To calculate accuracy
    
    # Create the loss object, optimizer, and training function
    loss_object = tf.keras.losses.MeanSquaredError()
    loss_object_adversarial = tf.losses.BinaryCrossentropy(from_logits=True) # More stable
    # from_logits = True because the last dense layers is linear and
    # does not have an activation -- could be differently specified
    
    optimizer = tf.keras.optimizers.Adam(0.01) # Possibly increase the learning rate to stir up the GAN training
    
    # 1. Start with embedder training (Optimal LSTM auto encoder network)
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), 
                                                dtype=tf.float64)])
    def train_step_embedder(X_train):
        with tf.GradientTape() as tape:
            # Apply Embedder to data and Recovery to predicted hidden states 
            e_pred_train = embedder_model(X_train)
            r_pred_train = recovery_model(e_pred_train)
            
            # Compute loss for LSTM autoencoder
            R_loss_train = loss_object(X_train, r_pred_train)
            #tf.debugging.assert_non_negative(r_loss_train) # Check if non-negative
        
        # Compute the gradients with respect to the Embedder and Recovery vars
        gradients = tape.gradient(R_loss_train, 
                                  embedder_model.trainable_variables +
                                  recovery_model.trainable_variables)
        
        # Apply the gradients to the Embedder and Recovery vars
        optimizer.apply_gradients(zip(gradients, # Always minimization function
                                      embedder_model.trainable_variables +
                                      recovery_model.trainable_variables))
        
        # Record the lower and upper layer gradients + the MSE for the autoencoder
        grad_embedder_ll(tf.norm(gradients[1]))
        grad_embedder_ul(tf.norm(gradients[9]))
        grad_recovery_ll(tf.norm(gradients[12]))
        grad_recovery_ul(tf.norm(gradients[20]))
        r_loss_train(R_loss_train)
        
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), 
                                                dtype=tf.float64)])
    def test_step_embedder(X_test):
        # Apply the Embedder to data and Recovery to predicted hidden states
        e_pred_test = embedder_model(X_test)
        r_pred_test = recovery_model(e_pred_test)
        
        # Compute the loss function for the LSTM autoencoder
        R_loss_test = loss_object(X_test, r_pred_test)
        r_loss_test(R_loss_test)    
    
    # Train the embedder for the input data
    for epoch in range(load_epochs, load_epochs + iterations):
        r_loss_train.reset_states()
        r_loss_test.reset_states()
       
        # Train over the complete train and test dataset
        for x_train in X_train:
            train_step_embedder(x_train)
       
        for x_test in X_test:
            test_step_embedder(x_test)
       
        with summary_writer_train.as_default():
            tf.summary.scalar('1. Pre-training autoencoder/1. loss', 
                              r_loss_train.result(), step=epoch)
            if epoch % 10 == 0: # Only log trainable variables per 10 epochs
                add_hist(embedder_model.trainable_variables, epoch)
                add_hist(recovery_model.trainable_variables, epoch)
        
        with summary_writer_test.as_default():
            tf.summary.scalar('1. Pre-training autoencoder/1. loss', 
                              r_loss_test.result(), step=epoch,
                              description = str(descr_auto_loss()))
        
        with summary_writer_bottom.as_default():
            tf.summary.scalar('1. Pre-training autoencoder/2. gradients - embedder',
                              grad_embedder_ll.result(), step=epoch)
            tf.summary.scalar('1. Pre-training autoencoder/2. gradients - recovery',
                              grad_recovery_ll.result(), step=epoch)
        
        with summary_writer_top.as_default():
            tf.summary.scalar('1. Pre-training autoencoder/2. gradients - embedder',
                              grad_embedder_ul.result(), step=epoch,
                              description = str(descr_auto_grads_embedder()))
            tf.summary.scalar('1. Pre-training autoencoder/2. gradients - recovery',
                              grad_recovery_ul.result(), step=epoch,
                              description = str(descr_auto_grads_recovery()))
       
        # Log the progress to the user console in python    
        template = 'Autoencoder training: Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch+1, 
                              np.round(r_loss_train.result().numpy(),5),
                              np.round(r_loss_test.result().numpy(), 5)))
   
    print('Finished Embedding Network Training')

    # 2. Continue w/ supervisor training on real data (same temporal relations)
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), 
                                                dtype=tf.float64)])
    def train_step_supervised(X_train):
      with tf.GradientTape() as tape:
        # Apply Embedder to data and check temporal relations with supervisor
        e_pred_train = embedder_model(X_train)
        H_hat_supervise = supervisor_model(e_pred_train)
        
        # Compute squared loss for real embedding and supervised embedding
        G_loss_S_train = loss_object(e_pred_train[:, 1:, :],
                               H_hat_supervise[:, 1:, :])
        tf.debugging.assert_non_negative(G_loss_S_train)
      
      # Compute the gradients with respect to the Embedder and Recovery vars
      gradients = tape.gradient(G_loss_S_train, 
                                supervisor_model.trainable_variables)
      
      # Apply the gradients to the Embedder and Recovery vars
      optimizer.apply_gradients(zip(gradients, # Always minimization
                                    supervisor_model.trainable_variables))
      
      # Record the lower and upper layer gradients + the MSE for the supervisor
      grad_supervisor_ll(tf.norm(gradients[1]))
      grad_supervisor_ul(tf.norm(gradients[6]))
      g_loss_s_train(G_loss_S_train)
      
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), 
                                                dtype=tf.float64)])
    def test_step_supervised(X_test):
        e_pred_test = embedder_model(X_test)
        H_hat_supervise_test = supervisor_model(e_pred_test)
        G_loss_S_test = loss_object(e_pred_test[:, 1:, :], 
                                    H_hat_supervise_test[:, 1:, :])
        g_loss_s_test(G_loss_S_test)
    
    for epoch in range(load_epochs, load_epochs + iterations):
        g_loss_s_train.reset_states()
        g_loss_s_test.reset_states()
        
        for x_train in X_train:
            train_step_supervised(x_train)
        
        for x_test in X_test:
            test_step_supervised(x_test)
        
        with summary_writer_train.as_default():
            tf.summary.scalar('2. Pre-training supervisor/1. loss', 
                              g_loss_s_train.result(), step=epoch)
            if epoch % 10 == 0: # Only log trainable variables per 10 epochs
                add_hist(supervisor_model.trainable_variables, epoch)
       
        with summary_writer_test.as_default():
                tf.summary.scalar('2. Pre-training supervisor/1. loss',
                              g_loss_s_test.result(), step=epoch,
                              description = str(descr_supervisor_loss()))
                
        with summary_writer_bottom.as_default():
            tf.summary.scalar('2. Pre-training supervisor/2. gradients - supervisor',
                              grad_supervisor_ll.result(), step=epoch)
        
        with summary_writer_top.as_default():
            tf.summary.scalar('2. Pre-training supervisor/2. gradients - supervisor',
                              grad_supervisor_ul.result(), step=epoch,
                              description = str(descr_auto_grads_supervisor()))
            
        template = 'Epoch {}, Train Loss: {}, Test loss: {}'
        print(template.format(epoch+1, 
                              np.round(g_loss_s_train.result().numpy(),8),
                              np.round(g_loss_s_test.result().numpy(),8) ) )
    print('Finished training with Supervised loss only')
    
    # 3. Continue with joint training
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), dtype=tf.float64),
                                  tf.TensorSpec(shape=(None,20,hidden_dim), dtype=tf.float64),
                                  tf.TensorSpec(shape=(), dtype = tf.bool)])
    def train_step_jointly_generator(X_train, Z, graphing=False):
        if graphing: # Only used for creating the graph
            with tf.GradientTape() as tape:
              # We need these steps to make the graph in Tensorboard complete
              dummy = embedder_model(x_train)
              dummy1 = recovery_model(dummy)
              dummy2 = supervisor_model(dummy)
              dummy3 = discriminator_model(dummy)
        else:
            with tf.GradientTape() as tape:
              # We need these steps to make the graph in Tensorboard complete
              dummy = embedder_model(x_train)
              dummy1 = recovery_model(dummy)
              dummy2 = supervisor_model(dummy)
                
              # Apply Generator to Z and apply Supervisor on fake embedding
              E_hat = generator_model(Z)
              H_hat = supervisor_model(E_hat)
                  
              # Compute real and fake probabilities using Discriminator model
              Y_fake_e = discriminator_model(E_hat)
              
              # 1. Generator - Adversarial loss - We want to trick Discriminator to give classification of 1
              G_loss_U_e = loss_object_adversarial(tf.ones_like(Y_fake_e), 
                                                   Y_fake_e)
              
              # 2. Generator - Supervised loss for fake embeddings
              G_loss_S = loss_object(E_hat[:, 1:, :], H_hat[:, 1:, :])
              
              # 3. Generator - 
              
              # Sum and multiply supervisor loss by eta for equal
              # contribution to generator loss function
              G_loss = G_loss_U_e + eta * G_loss_S  
              
            # Compute the gradients w.r.t. generator and supervisor model
            gradients_generator=tape.gradient(G_loss,
                                              generator_model.trainable_variables)
            
            # Apply the gradients to the generator model
            optimizer.apply_gradients(zip(gradients_generator, 
                                          generator_model.trainable_variables)) 
        
            # Record the lower and upper layer gradients + the MSE for the generator
            grad_generator_ll(tf.norm(gradients_generator[1]))
            grad_generator_ul(tf.norm(gradients_generator[9]))
            
            # Compute individual components of the generator loss
            g_loss_u_e(G_loss_U_e)
            g_loss_s(G_loss_S) # Based on this we can set the eta value in G_loss_S
     
    # 3. Continue with joint training
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), dtype=tf.float64),
                                  tf.TensorSpec(shape=(None,20,hidden_dim), dtype=tf.float64)])
    def train_step_jointly_generator(X_train, Z):
        # Apply Generator to Z and apply Supervisor on fake embedding
        E_hat = generator_model(Z)
        H_hat = supervisor_model(E_hat)
            
        # Compute real and fake probabilities using Discriminator model
        Y_fake_e = discriminator_model(E_hat)
        
        # 1. Generator - Adversarial loss - We want to trick Discriminator to give classification of 1
        G_loss_U_e_test = loss_object_adversarial(tf.ones_like(Y_fake_e), 
                                             Y_fake_e)
        
        # 2. Generator - Supervised loss for fake embeddings
        G_loss_S_test = loss_object(E_hat[:, 1:, :], H_hat[:, 1:, :])
        
        # 3. Generator - 
          
        # Log individual components of generator loss
        g_loss_u_e_test(G_loss_U_e_test)
        g_loss_s_test(G_loss_S_test)        
                
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), dtype=tf.float64)])
    def train_step_jointly_embedder(X_train):
        with tf.GradientTape() as tape:
          # Apply Embedder to data and recover the data from the embedding space
          H = embedder_model(X_train) 
          X_tilde = recovery_model(H)
          
          # Compute the loss function for the embedder-recovery model
          r_loss_train = loss_object(X_train, X_tilde)  
          
          # Include the supervision loss but only for 10 %
          H_hat_supervise = supervisor_model(H)
          G_loss_S_embedder = loss_object(H[:,1:,:], H_hat_supervise[:,1:,:])
          
          # Combine the two losses
          E_loss = r_loss_train + lambda_val * tf.sqrt(G_loss_S_embedder)
          
          # Compute the gradients with respect to the embedder-recovery model
          gradients_embedder=tape.gradient(E_loss,
                                           embedder_model.trainable_variables + 
                                           recovery_model.trainable_variables)
         
          optimizer.apply_gradients(zip(gradients_embedder,
                                        embedder_model.trainable_variables + 
                                        recovery_model.trainable_variables))
    
        # Compute the embedding-recovery loss and supervisor loss
        e_loss_T0(r_loss_train) 
        g_loss_s_embedder(G_loss_S_embedder)
        
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), dtype=tf.float64)])
    def test_step_jointly_embedder(X_test):
      # Apply Embedder on test and make reconstruction from embedding space H
      H = embedder_model(X_test) 
      X_tilde = recovery_model(H)
      
      # Compute the loss function for the embedder-recovery model
      R_loss_test = loss_object(X_train, X_tilde)  
      
      # Include the supervision loss but only for 10 %
      H_hat_supervise = supervisor_model(H)
      G_loss_S_embedder_test = loss_object(H[:,1:,:], H_hat_supervise[:,1:,:])

      # Compute the embedding-recovery loss and supervisor loss
      e_loss_T0_test(R_loss_test) 
      g_loss_s_embedder_test(G_loss_S_embedder_test)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), dtype=tf.float64),
                                  tf.TensorSpec(shape=(None,20, hidden_dim), dtype=tf.float64),
                                  tf.TensorSpec(shape=(), dtype = tf.float64)])
    def train_step_discriminator(X_train, Z, smoothing_factor=1):
        with tf.GradientTape() as tape:
            # Embeddings for real data and classifications from discriminator
            H = embedder_model(X_train)
            Y_real = discriminator_model(H) 
            
            # Embeddings for fake data and classifications from discriminator
            E_hat = generator_model(Z)
            Y_fake_e = discriminator_model(E_hat)
            
            # Loss for the discriminator
            D_loss_real = loss_object_adversarial(tf.ones_like(Y_real) * smoothing_factor, Y_real)
            D_loss_fake_e = loss_object_adversarial(tf.zeros_like(Y_fake_e), 
                                                    Y_fake_e)
            D_loss = D_loss_real + D_loss_fake_e
            D_loss = -D_loss
            
        # Compute the gradients with respect to the discriminator model
        grad_d=tape.gradient(D_loss,
                             discriminator_model.trainable_variables)
            
        # Apply the gradient to the discriminator model
        optimizer.apply_gradients(zip(grad_d, # Minimize the Cross Entropy
                                      discriminator_model.trainable_variables))
        
        # Record the lower and upper layer gradients + the MSE for the discriminator
        grad_discriminator_ll(tf.norm(grad_d[1]))
        grad_discriminator_ul(tf.norm(grad_d[9]))
        d_loss(D_loss)
     
    def evaluate_accuracy(X_test, Z):
        Y_real_test = (discriminator_model.predict(embedder_model(X_test)).numpy() > 0.5) * 1
        Y_fake_test = (discriminator_model.predict(Z).numpy() > 0.5) * 1
        
        # Compute the loss 
        D_accuracy_real = loss_object_accuracy(tf.ones_like(Y_real_test), 
                                               Y_real_test).numpy()
        D_accuracy_fake = loss_object_accuracy(tf.zeros_like(Y_fake_test), 
                                               Y_fake_test).numpy()
        
        return D_accuracy_real, D_accuracy_fake
    
    # Helper counter for the already performed epochs
    already_done_epochs = epoch
        
    # Define the algorithm for training jointly
    print('Start joint training')
    o = 0
    for epoch in range(load_epochs, iterations+load_epochs):
        g_loss_u_e.reset_states() # Reset the loss at every epoch
        g_loss_s.reset_states()
        e_loss_T0.reset_states()
        g_loss_s_embedder.reset_states()
        d_loss.reset_states()
        
        # This for loop is GENERATOR TRAINING
        # Create 1 generator and embedding training iters.
        for kk in range(1): # Make a random generation
            if epoch == 0 and kk == 1:
                # Train the generator and embedder sequentially
                for x_train in X_train:
                    Z_mb = RandomGenerator(batch_size, [20, hidden_dim])
                    train_step_jointly_generator(x_train, Z_mb, tf.constant(True, dtype=tf.bool))
                    train_step_jointly_embedder(x_train)
            else:
                # Train the generator and embedder sequentially
                for x_train in X_train:
                    Z_mb = RandomGenerator(batch_size, [20, hidden_dim])
                    train_step_jointly_generator(x_train, Z_mb, tf.constant(False, dtype=tf.bool))
                    train_step_jointly_embedder(x_train)
                    train_step_jointly_embedder(x_train) # Double embedder training for better embedding space
       
        # This for loop is DISCRIMINATOR TRAINING
        # Train discriminator if too bad or at initialization (0.0)
        if d_loss.result() > 0.15 or d_loss.result() == 0.0:
            current_o = o
            while d_loss.result() > 0.15 and o < current_o + 5: # Train d to optimum (Jensen-Shannon divergence)
                for x_train in X_train: # Train the discriminator for a maximum of 10 iterations or stop if optimal discriminator
                    Z_mb = RandomGenerator(batch_size, [20, hidden_dim])
                    train_step_discriminator(x_train, Z_mb, smoothing_factor = 1.0)
                    with summary_writer_train.as_default():
                        tf.summary.scalar('3. TimeGAN training - GAN/1. GAN loss', 
                                          g_loss_s.result() + d_loss.result(), step=o, description = str(descr_discriminator_loss_joint()))
                    with summary_writer_lower_bound.as_default():
                        tf.summary.scalar('3. TimeGAN training - GAN/1. GAN loss',
                                          -2*np.log(2), step=o, description = 'GAN loss with lower bound.')
                    o += 1    
                    print(d_loss.result().numpy())
        
        # Compute the test accuracy
        acc_real_array = np.array([])
        acc_fake_array = np.array([])
        
        for x_test in X_test:
            Z_mb = RandomGenerator(batch_size, [20, hidden_dim])
            acc_real, acc_fake = evaluate_accuracy(x_test, Z_mb)
            
            acc_real_array = np.append(acc_real_array, acc_real)
            acc_fake_array = np.append(acc_fake_array, acc_fake)
            
        with summary_writer_train.as_default():  
            # Log autoencoder + supervisor losses
            tf.summary.scalar('3. TimeGAN training - Autoencoder/1. loss - recovery', 
                              e_loss_T0.result(), step=epoch,
                              description = str(descr_auto_loss_joint_auto()))
            tf.summary.scalar('3. TimeGAN training - Autoencoder/1. loss - supervisor',
                              g_loss_s_embedder.result(), step=epoch,
                              description = str(descr_auto_loss_joint_supervisor()))
            
            # Log GAN + supervisor losses
            tf.summary.scalar('3. TimeGAN training - GAN/1. loss - GAN',
                              d_loss.result(), step=epoch,
                              description = str(descr_generator_loss_joint()))
            tf.summary.scalar('3. TimeGAN training - GAN/1. loss - supervisor',
                              g_loss_s.result(), step=epoch,
                              description = str(descr_supervisor_loss_joint()))
            
        with summary_writer_test.as_default():
            # Log autoencoder + supervisor losses
            tf.summary.scalar('3. TimeGAN training - Autoencoder/1. loss - recovery',
                              e_loss_T0_test.result(), step=epoch)
            
            tf.summary.scalar('TimeGAN training - Autoencoder/1. loss - supervisor',
                              g_loss_s_embedder_test.result())
       
        with summary_writer_real_data.as_default():   
            tf.summary.scalar('3. TimeGAN training - GAN/2. Accuracy',
                              tf.reduce_mean(acc_real_array), step = epoch)
        
        with summary_writer_fake_data.as_default():
            tf.summary.scalar('3. TimeGAN training - GAN/2. Accuracy',
                              tf.reduce_mean(acc_fake_array), step = epoch,
                              description = str(descr_accuracy_joint()))
        
        with summary_writer_bottom.as_default():
            tf.summary.scalar('3. TimeGAN training - GAN/3. gradients - discriminator',
                              grad_discriminator_ll.result(), step=epoch)
            tf.summary.scalar('3. TimeGAN training - GAN/3. gradients - generator',
                              grad_generator_ll.result(), step=epoch)
        
        with summary_writer_top.as_default():
            tf.summary.scalar('3. TimeGAN training - GAN/3. gradients - discriminator',
                              grad_discriminator_ul.result(), step=epoch,
                              description = str(descr_joint_grad_discriminator()))
                              
            tf.summary.scalar('3. TimeGAN training - GAN/3. gradients - generator',
                              grad_generator_ul.result(), step=epoch,
                              description = str(descr_joint_grad_generator()))
        
        # with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        #     tf.summary.scalar(METRIC_G_LOSS_U, g_loss_u.result(), step=epoch)
        #     tf.summary.scalar(METRIC_G_LOSS_V, g_loss_v.result(), step=epoch)
        #     tf.summary.scalar(METRIC_G_LOSS_S, g_loss_s.result(), step=epoch)
        #     tf.summary.scalar(METRIC_E_LOSS, e_loss_T0.result(), step=epoch)
        #     tf.summary.scalar(METRIC_D_LOSS, d_loss.result(), step=epoch)
        
            # Only log the weights of the model per 10 epochs
            if epoch % 10 == 0:  # Add variables to histogram and distribution   
                
                # Pre-trained models
                add_hist(embedder_model.trainable_variables,
                         epoch + already_done_epochs) 
                add_hist(recovery_model.trainable_variables,
                         epoch + already_done_epochs) 
                add_hist(supervisor_model.trainable_variables,
                         epoch + already_done_epochs)
                
                # Not pre-trained models
                add_hist(generator_model.trainable_variables, epoch) 
                add_hist(discriminator_model.trainable_variables, epoch)
                
            if epoch % 50 == 0: # It takes around an hour to do 10 epochs
                # Lastly save all models
                embedder_model.save_weights('weights/embedder/epoch_' + str(epoch))
                recovery_model.save_weights('weights/recovery/epoch_' + str(epoch))
                supervisor_model.save_weights('weights/supervisor/epoch_' + str(epoch))
                generator_model.save_weights('weights/generator/epoch_' + str(epoch))
                discriminator_model.save_weights('weights/discriminator/epoch_' + str(epoch)) 

                # Convert the model into interpretable simulations and Nearest-Neighbour comparisons
                figure = image_grid(1000, 20, 4, recovery_model, generator_model)
                
                figure.canvas.draw()
                w, h = figure.canvas.get_width_height()
                img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape((1, h, w, 3))   
                
                with summary_writer_train.as_default():
                    tensor = tf.constant(img)
                    tf.summary.image(str("Simulations & nearest neighbour after " + str(epoch) + " training iterations"),
                                     tensor, step=epoch, 
                                     description = str(descr_images()))

        
            print('step: '+ str(epoch+1) +  
                  ', g_loss_u_e: ' + str(np.round(g_loss_u_e.result().numpy(),8)) + 
                  ', g_loss_s: ' + str(np.round(g_loss_s.result().numpy(),8)) +                   
                  ', g_loss_s_embedder: ' + 
                  str(np.round(g_loss_s_embedder.result().numpy(),8)) +
                  ', e_loss_t0: ' + str(np.round(e_loss_T0.result().numpy(),8)) + 
                  ', d_loss: ' + str(np.round(d_loss.result().numpy(),8))) 
            
    print('Finish joint training')