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

# 3. Train TimeGAN model
from models.Embedder import Embedder
from models.Recovery import Recovery
from models.Generator import Generator
from models.Supervisor import Supervisor
from models.Discriminator import Discriminator
from training import add_hist, RandomGenerator

os.chdir('C://Users/s157148/Documents/Github/TimeGAN')

def run(parameters, hparams, X_train, X_test):
    
    # Network Parameters
    hidden_dim   = parameters['hidden_dim']  #
    num_layers   = parameters['num_layers']  # Still have to implement
    iterations   = parameters['iterations']  # Test run to check for overfitting
    batch_size   = parameters['batch_size']  # Currently locked at 25
    module_name  = parameters['module_name'] # 'lstm' or 'GRU''
    z_dim        = parameters['z_dim']       # Currently locked at 5
    lambda_val   = 6
    eta          = 75
    
    # Define the TensorBoard such that we can visualize the results
    log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer_train = tf.summary.create_file_writer(log_dir + '/train')
    summary_writer_test = tf.summary.create_file_writer(log_dir + '/test')
    
    # Create an instance of all neural networks models (All LSTM)
    embedder_model = Embedder('logs/embedder', hparams, dimensionality = 1)
    recovery_model = Recovery('logs/recovery', hparams, dimensionality = 1) # If used for EONIA rate only
    generator_model = Generator('logs/generator', hparams)
    supervisor_model = Supervisor('logs/supervisor', hparams)
    discriminator_model = Discriminator('logs/TimeGAN', hparams)
    
    r_loss_train = tf.keras.metrics.Mean(name='r_loss_train') # Step 1 metrics 
    r_loss_test = tf.keras.metrics.Mean(name='r_loss_test')
    
    g_loss_s_train = tf.keras.metrics.Mean(name='g_loss_s_train') # Step 2 metrics
    g_loss_s_test = tf.keras.metrics.Mean(name='g_loss_s_test')
    
    e_loss_T0 = tf.keras.metrics.Mean(name='e_loss_T0') # Step 3 metrics
    g_loss_s_embedder = tf.keras.metrics.Mean(name='g_loss_s_embedder')
    g_loss_s = tf.keras.metrics.Mean(name='g_loss_s')
    d_loss = tf.keras.metrics.Mean(name='d_loss')
    g_loss_u_e = tf.keras.metrics.Mean(name='g_loss_u_e')
    
    # Create the loss object, optimizer, and training function
    loss_object = tf.keras.losses.MeanSquaredError()
    loss_object_adversarial = tf.losses.BinaryCrossentropy(from_logits=True)
    # from_logits = True because the last dense layers is linear and
    # does not have an activation -- could be differently specified
    optimizer = tf.keras.optimizers.Adam(0.001)
    
    #tf.compat.v1.disable_eager_execution()
    
    # 1. Start with embedder training (Optimal LSTM auto encoder network)
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), 
                                                dtype=tf.float64)])
    def train_step_embedder(X_train):
        with tf.GradientTape() as tape:
            # Apply Embedder to data and Recovery to predicted hidden states 
            e_pred_train = embedder_model(X_train)
            
            #print(e_pred_train) # Only printed while tracing the graph in first run
            r_pred_train = recovery_model(e_pred_train)
            #print(r_pred_train) # Only printed while tracing the graph in first run
            
            # Compute loss for LSTM autoencoder
            R_loss_train = loss_object(X_train, r_pred_train)
            #print(R_loss_train) # Only printed while tracing the graph in the first run
            #tf.debugging.assert_non_negative(r_loss_train) # Check if non-negative
        
        # Compute the gradients with respect to the Embedder and Recovery vars
        gradients = tape.gradient(R_loss_train, 
                                  embedder_model.trainable_variables +
                                  recovery_model.trainable_variables)
        
        # Apply the gradients to the Embedder and Recovery vars
        optimizer.apply_gradients(zip(gradients, 
                                      embedder_model.trainable_variables +
                                      recovery_model.trainable_variables))
        
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
    for epoch in range(iterations):
        r_loss_train.reset_states()
        r_loss_test.reset_states()
       
        # Train over the complete train and test dataset
        for x_train in X_train:
            train_step_embedder(x_train)
       
        for x_test in X_test:
            test_step_embedder(x_test)
       
        with summary_writer_train.as_default():
            tf.summary.scalar('1. Autoencoder training/', 
                              r_loss_train.result(),step=epoch)
            if epoch % 10 == 0: # Only log trainable variables per 10 epochs
                add_hist(embedder_model.trainable_variables, epoch)
                add_hist(recovery_model.trainable_variables, epoch)
        
        with summary_writer_test.as_default():
            tf.summary.scalar('1. Autoencoder training/', 
                              r_loss_test.result(), step=epoch)
       
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
      optimizer.apply_gradients(zip(gradients, 
                                    supervisor_model.trainable_variables))
      
      # Compute the training loss for the supervised model
      g_loss_s_train(G_loss_S_train)
      
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), 
                                                dtype=tf.float64)])
    def test_step_supervised(X_test):
        e_pred_test = embedder_model(X_test)
        H_hat_supervise_test = supervisor_model(e_pred_test)
        G_loss_S_test = loss_object(e_pred_test[:, 1:, :], 
                                    H_hat_supervise_test[:, 1:, :])
        g_loss_s_test(G_loss_S_test)
    
    for epoch in range(iterations):
        g_loss_s_train.reset_states()
        g_loss_s_test.reset_states()
        
        for x_train in X_train:
            train_step_supervised(x_train)
        
        for x_test in X_test:
            test_step_supervised(x_test)
        
        with summary_writer_train.as_default():
            tf.summary.scalar('2. Temporal relations training/', 
                              g_loss_s_train.result(), step=epoch)
            if epoch % 10 == 0: # Only log trainable variables per 10 epochs
                add_hist(generator_model.trainable_variables, epoch)
                add_hist(supervisor_model.trainable_variables, epoch)
        with summary_writer_test.as_default():
                tf.summary.scalar('2. Temporal relations training/',
                              g_loss_s_test.result(), step=epoch)
            
        template = 'Epoch {}, Train Loss: {}, Test loss: {}'
        print(template.format(epoch+1, 
                              np.round(g_loss_s_train.result().numpy(),8),
                              np.round(g_loss_s_test.result().numpy(),8) ) )
    print('Finished training with Supervised loss only')
    
    # 3. Continue with joint training
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), dtype=tf.float64),
                                  tf.TensorSpec(shape=(None,20,hidden_dim), dtype=tf.float64)])
    def train_step_jointly_generator(X_train, Z):
        with tf.GradientTape() as tape:
          # Apply Generator to Z and apply Supervisor on fake embedding
          E_hat = generator_model(Z)
          H_hat = supervisor_model(E_hat)
          
          # Compute real and fake probabilities using Discriminator model
          Y_fake_e = discriminator_model(E_hat)
          
          # 1. Generator - Adversarial loss - We want classification to be 1
          G_loss_U_e = loss_object_adversarial(tf.ones_like(Y_fake_e), 
                                               Y_fake_e)
          
          # 2. Generator - Supervised loss for fake embeddings
          G_loss_S = loss_object(E_hat[:, 1:, :], H_hat[:, 1:, :])
          
          # Sum and multiply supervisor loss by eta for equal
          # contribution to generator loss function
          G_loss = G_loss_U_e + eta * tf.sqrt(G_loss_S)  
          
        # Compute the gradients w.r.t. generator and supervisor model
        gradients_generator=tape.gradient(G_loss,
                                          generator_model.trainable_variables)
        
        # Apply the gradients to the generator model
        optimizer.apply_gradients(zip(gradients_generator, 
                                      generator_model.trainable_variables)) 
    
        # Compute individual components of the generator loss
        g_loss_u_e(G_loss_U_e)
        g_loss_s(G_loss_S) # Based on this we can set the eta value in G_loss_S
    
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
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,20,1), dtype=tf.float64),
                                  tf.TensorSpec(shape=(None,20, hidden_dim), dtype=tf.float64)])
    def train_step_discriminator(X_train, Z):
        with tf.GradientTape() as tape:
            # Embeddings for real data and classifications from discriminator
            H = embedder_model(X_train)
            Y_real = discriminator_model(H) 
            
            # Embeddings for fake data and classifications from discriminator
            E_hat = generator_model(Z)
            Y_fake_e = discriminator_model(E_hat)
            
            # Loss for the discriminator
            D_loss_real = loss_object_adversarial(tf.ones_like(Y_real), Y_real)
            D_loss_fake_e = loss_object_adversarial(tf.zeros_like(Y_fake_e), 
                                                    Y_fake_e)
            D_loss = D_loss_real + D_loss_fake_e
            
            # Compute the gradients with respect to the discriminator model
            grad_d=tape.gradient(D_loss,
                                 discriminator_model.trainable_variables)
                
            # Apply the gradient to the discriminator model
            optimizer.apply_gradients(zip(grad_d,
                                          discriminator_model.trainable_variables))
                
            # Compute the discriminator loss
            d_loss(D_loss) 
        
    # Helper counter for the already performed epochs
    already_done_epochs = epoch
        
    # Define the algorithm for training jointly
    print('Start joint training')
    for epoch in range(iterations):
        g_loss_u_e.reset_states() # Reset the loss at every epoch
        g_loss_s.reset_states()
        e_loss_T0.reset_states()
        g_loss_s_embedder.reset_states()
        
        d_loss.reset_states()
        
        # This for loop is the generator training
        # Create 2 generator and embedding training iters. Just like paper.
        for kk in range(2): # Make a random generation
            Z_minibatch = RandomGenerator(batch_size, [20, hidden_dim])
                
            # Train the generator and embedder sequentially
            for x_train in X_train:
                train_step_jointly_generator(x_train, Z_minibatch)
                train_step_jointly_embedder(x_train)
       
        # This for loop is the discriminator training
        # Train discriminator if too bad or at initialization (0.0)
        if d_loss.result() > 0.15 or d_loss.result() == 0.0:
            for kk in range(7): # Train d to optimum (Jensen-Shannon diverge)
                Z_mb = RandomGenerator(batch_size, [20, hidden_dim])
                for x_train in X_train: # Train the discriminator for 5 epochs
                    train_step_discriminator(x_train, Z_mb)
            
        with summary_writer_train.as_default():  
            # Log autoencoder + supervisor losses
            tf.summary.scalar('3. Joint training - Autoencoder/recovery', 
                              e_loss_T0.result(), step=epoch)
            tf.summary.scalar('3. Joint training - Autoencoder/supervisor',
                              g_loss_s_embedder.result(), step=epoch)
            
            # Log supervisor loss on fake embedding samples
            tf.summary.scalar('3. Joint training - Temporal relations/supervisor',
                              g_loss_s.result(), step=epoch)
        
            # Log GAN losses
            tf.summary.scalar('3. Joint training - GAN/generator',
                              g_loss_u_e.result(), step=epoch)
            tf.summary.scalar('3. Joint training - GAN/discriminator', 
                              d_loss.result(), step=epoch)
            
        # with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        #     tf.summary.scalar(METRIC_G_LOSS_U, g_loss_u.result(), step=epoch)
        #     tf.summary.scalar(METRIC_G_LOSS_V, g_loss_v.result(), step=epoch)
        #     tf.summary.scalar(METRIC_G_LOSS_S, g_loss_s.result(), step=epoch)
        #     tf.summary.scalar(METRIC_E_LOSS, e_loss_T0.result(), step=epoch)
        #     tf.summary.scalar(METRIC_D_LOSS, d_loss.result(), step=epoch)
        
            # Only log the weights of the model per 10 epochs
            if epoch % 10 == 0:    
                # Add all variables to the histogram and distribution 
                add_hist(generator_model.trainable_variables, 
                         epoch + already_done_epochs)
                add_hist(supervisor_model.trainable_variables,
                         epoch + already_done_epochs)
                add_hist(embedder_model.trainable_variables,
                         epoch + already_done_epochs)
                add_hist(discriminator_model.trainable_variables,
                         epoch + already_done_epochs)
                add_hist(supervisor_model.trainable_variables,
                         epoch + already_done_epochs)
                
            if epoch % 50 == 0: # It takes around an hour to do 10 epochs
                # Lastly save all models
                embedder_model.save_weights('weights/embedder/epoch_' + str(epoch))
                recovery_model.save_weights('weights/recovery/epoch_' + str(epoch))
                supervisor_model.save_weights('weights/supervisor/epoch_' + str(epoch))
                generator_model.save_weights('weights/generator/epoch_' + str(epoch))
                discriminator_model.save_weights('weights/discriminator/epoch_' + str(epoch)) 
        
            print('step: '+ str(epoch+1) +  
                  ', g_loss_u_e: ' + str(np.round(g_loss_u_e.result().numpy(),8)) + 
                  ', g_loss_s: ' + str(np.round(np.sqrt(g_loss_s.result().numpy()),8)) + 
                  #', g_loss_v: ' + str(np.round(g_loss_v.result().numpy(),8)) + 
                  
                  ', g_loss_s_embedder: ' + 
                  str(np.round(g_loss_s_embedder.result().numpy(),8)) +
                  ', e_loss_t0: ' + str(np.round(np.sqrt(e_loss_T0.result().numpy()),8)) + 
                  ', d_loss: ' + str(np.round(d_loss.result().numpy(),8))) 
            
        print('Finish joint training')
