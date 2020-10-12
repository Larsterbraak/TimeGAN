"""
MSc Thesis Quantitative Finance - Erasmus University Rotterdam
Title: Interest rate risk simulation using TimeGAN after the EONIA-ESTER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: Oct 7th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Inputs
(1) EONIA, pre-ESTER & ESTER dataset
(2) Network architecture
- Which loss function?
- Positive label smoothing?
- Feature matching?
(3) Hyperparameters
- Supervised loss w.r.t. Autoencoder loss
- Supervised loss w.r.t Unsupervised loss
- Feature matching loss
- Gradient penalty loss
(4) Pre-trained models

Outputs
(1) Log files containing containing Tensorboard visualizations
(2) Log files containing the weights of trained models
"""

# Import generic Python packages
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras import backend as K

# Import the self-defined models used in TimeGAN
from models.Embedder import Embedder
from models.Recovery import Recovery
from models.Generator import Generator
from models.Supervisor import Supervisor
from models.Discriminator import Discriminator

# Import the self-defined helper scripts
from training import add_hist, RandomGenerator
from metrics import load_models

# Import the descriptions for the different Tensorboard tabs
from descriptions_tensorboard import (descr_pretrain_auto_loss, descr_pretrain_auto_embedder_grads,
                                      descr_pretrain_auto_recovery_grads, descr_pretrain_supervisor_loss,
                                      descr_pretrain_supervisor_grads, descr_joint_auto_loss, 
                                      descr_joint_supervisor_loss, descr_joint_generator_loss_wasserstein,
                                      descr_joint_generator_loss, descr_joint_generator_grads,
                                      descr_joint_discriminator_loss_wasserstein,
                                      descr_joint_discriminator_loss, descr_joint_discriminator_grads,
                                      descr_joint_accuracy, descr_joint_feature_matching_loss,
                                      descr_joint_gradient_penalty)

# Training on a multi-GPU core on a Linux server using a mirrored strategy approach
# Initialize the Mirrored strategy approach in Tensorflow 2.0
mirrored_strategy = tf.distribute.MirroredStrategy()

def run(parameters, hparams, X_train, X_test, 
        load=False, load_epochs=0, load_log_dir=""):
    """Train the TimeGAN model on the training dataset and evaluate on the test dataset.
    All results are logged to a TensorFlow 2.0 Tensorboard."""
    
    # Network Parameters
    hidden_dim   = parameters['hidden_dim'] # The dimensionality for the latent space
    num_layers   = parameters['num_layers'] # The number of layers in the network architectures
    iterations   = parameters['iterations'] # The number of iterations for jointly training
    batch_size   = parameters['batch_size'] # The batch size used during training
    module_name  = parameters['module_name'] # (Could also implement GRU and LSTM layer normalization)
    z_dim        = parameters['z_dim'] # The dimensionality of the randomly generated input       
    
    # Hyperparameters
    lambda_val   = 1.0 # Hyperparameter for supervised loss w.r.t recovery loss
    eta          = 1.0 # Hyperparameter for supervised loss w.r.t unsupervised loss
    kappa        = 50.0 # Hyperparameter for feature matching loss
    gamma        = 0.1  # Hyperparameter for gradient penalty (based on (Gulrajani et al., 2017))
    
    # Load models or make new run?  
    if load:
        log_dir = load_log_dir
    else:
        today = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Log directories for Normal, Feature Matching, Positive Label Smoothing and WGAN-GP training
        # log_dir = 'logs/Normal_1_disc_step_0_05_gen_0_0008_disc_' + today
        # log_dir = 'logs/FM,lrG-0.02,lrD-0.0008,minmax,' + today
        # log_dir = 'logs/PLM,lrG-0.02,lrD-0.0008,minmax,' + today
        # log_dir = 'logs/ALL,lrG-0.02,lrD-0.0008,minmax,' + today
        log_dir = 'logs/WGANGP5Dsteps,lrG-0.0005,lrD-0.0005,minmax,gamma0.01,oldnormcalc,withFM,1e-4forsqrt,' + today
    
    # Initialize the summary writers for the Tensorboard (TF visualization tool)
    # 1. Performance on train and test data set
    summary_writer_train = tf.summary.create_file_writer(log_dir + '/train')
    summary_writer_test = tf.summary.create_file_writer(log_dir + '/test')
    
    # 2. Gradients during all GAN training (To check for instability in log loss)
    summary_writer_bottom = tf.summary.create_file_writer(log_dir + '/bottom')
    summary_writer_top = tf.summary.create_file_writer(log_dir + '/top')

    # 3. Accuracy metric only for Normal GAN training (Goodfellow et al., 2014)
    summary_writer_real_data = tf.summary.create_file_writer(log_dir + '/real_data')
    summary_writer_fake_data = tf.summary.create_file_writer(log_dir + '/fake_data')
    
    # 4. Gradient penalty only for Wasserstein Gradient Penalty (Gulrajani et al., 2017)
    summary_writer_gp = tf.summary.create_file_writer(log_dir + '/gp')
    summary_writer_gp_test = tf.summary.create_file_writer(log_dir + '/gp_test')
    
    # Load pre-trained models or initialize all TimeGAN models
    if load:
        embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(load_epochs)
    else:
        with mirrored_strategy.scope():
            # Create an instance of all neural networks models
            embedder_model = Embedder('logs/embedder', hparams, hidden_dim, dimensionality = 11)
            recovery_model = Recovery('logs/recovery', hparams, hidden_dim, dimensionality = 11) 
            supervisor_model = Supervisor('logs/supervisor', hparams, hidden_dim)
            generator_model = Generator('logs/generator', hparams, hidden_dim)
            discriminator_model = Discriminator('logs/TimeGAN', hparams, hidden_dim)
        
    # Initiliaze the metrics to keep track off during all training steps of TimeGAN
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
    d_loss_real = tf.keras.metrics.Mean(name='d_loss_real')
    d_loss_fake = tf.keras.metrics.Mean(name='d_loss_fake')
    g_loss_u_e = tf.keras.metrics.Mean(name='g_loss_u_e')
    g_loss_v1 = tf.keras.metrics.Mean(name='g_loss_v1')
    w_loss_gp = tf.keras.metrics.Mean(name='w_loss_gp')
    w_loss_gp_test = tf.keras.metrics.Mean(name='w_loss_gp_test')

    e_loss_T0_test = tf.keras.metrics.Mean(name='e_loss_T0_test') # Step 3 metrics (test)
    g_loss_s_embedder_test = tf.keras.metrics.Mean(name='e_loss_T0_test')
    g_loss_s_test = tf.keras.metrics.Mean(name='g_loss_s_test')
    g_loss_u_e_test = tf.keras.metrics.Mean(name='g_loss_u_e_test')
    d_loss_test = tf.keras.metrics.Mean(name='d_loss_test')

    grad_discriminator_ll = tf.keras.metrics.Mean(name='d_grad_lower_layer') # Step 3 gradients
    grad_discriminator_ul = tf.keras.metrics.Mean(name='d_grad_upper_layer')
    grad_generator_ll = tf.keras.metrics.Mean(name='g_grad_lower_layer')
    grad_generator_ul = tf.keras.metrics.Mean(name='g_grad_upper_layer')
    
    # Create the loss objects to evaluate during training
    loss_object_accuracy = tf.keras.metrics.Accuracy() # To calculate accuracy
    loss_object_mse = tf.keras.losses.MeanSquaredError( # To calculate MSE
                      reduction=tf.keras.losses.Reduction.NONE) 
    loss_object_adversarial = tf.losses.BinaryCrossentropy( # To calculate the log-loss
                              from_logits=True, # More stable to evaluate from logits
                              reduction=tf.keras.losses.Reduction.NONE)

    # Initialize the optimizers using the Mirrored Strategy approach
    with mirrored_strategy.scope():
        # Momentum based optimizers when using log loss (Goodfellow et al., 2014)
        # optimizer_discriminator = tf.keras.optimizers.Adam(0.0008, 0.5) # Less momentum in discriminator
        # optimizer_generator = tf.keras.optimizers.Adam(0.02)
        
        # No momentum based optimizers when using Wasserstein-1 loss (Arjovsky et al., 2017)
        # optimizer_discriminator = tf.keras.optimizers.RMSprop(0.00005)

        # Momentum based optimizer when using Wasserstein-1 loss with gradient penalty (Gulrajani et al., 2017)
        optimizer_discriminator = tf.keras.optimizers.Adam(0.0005, 0.5) # Somewhat less momentum than standard
        optimizer_generator = tf.keras.optimizers.RMSprop(0.0005)
        
        # High learning rate for Autoencoder and supervisor networks
        optimizer_embedder = tf.keras.optimizers.Adam(0.01)
        optimizer_supervisor = tf.keras.optimizers.Adam(0.05)

    # Change the input datasets to be used by the mirrored strategy
    X_train = mirrored_strategy.experimental_distribute_dataset(X_train)
    X_test = mirrored_strategy.experimental_distribute_dataset(X_test)
    
    # Compute the MSE loss according to the MirroredStrategy approach
    def compute_mse_loss(real, regenerate):
        per_example_loss = loss_object_mse(real, regenerate)
        return tf.nn.compute_average_loss(per_example_loss, 
                                          global_batch_size=batch_size * mirrored_strategy.num_replicas_in_sync)
     
    # Compute the wasserstein loss according to the MirroredStrategy approach
    def compute_wasserstein_loss(real, regenerate, generator=False, training=True):
        if generator:            
            per_example_loss = - discriminator_model(regenerate, training)
        else:
            per_example_loss = discriminator_model(regenerate, training) - discriminator_model(real, training)
        return tf.nn.compute_average_loss(per_example_loss,
                                          global_batch_size = batch_size * mirrored_strategy.num_replicas_in_sync)
    
    # Compute the Binary Cross entropy according to the MirroredStrategy approach
    def compute_adversarial_loss(real, regenerate, generator=False, visualization=False):
        if generator:
          per_example_loss = loss_object_adversarial(real, regenerate)
        elif visualization:
          per_example_loss = loss_object_adversarial(real, regenerate)
        else:
          per_example_loss = loss_object_adversarial(real, regenerate)
        return tf.nn.compute_average_loss(per_example_loss,
                                          global_batch_size = batch_size * mirrored_strategy.num_replicas_in_sync)
    
    # 1. Autoencoder training
    def train_step_embedder(X_train):
        with tf.GradientTape() as tape:
            # Apply Embedder to data and Recovery to predicted hidden states 
            e_pred_train = embedder_model(X_train, training=True)
            r_pred_train = recovery_model(e_pred_train, training=True)
            
            # Compute the loss for the LSTM autoencoder using MirroredStrategy
            R_loss_train = compute_mse_loss(X_train, r_pred_train)
            
        # Compute the gradients with respect to the Embedder and Recovery vars
        gradients = tape.gradient(R_loss_train, 
                                  embedder_model.trainable_variables +
                                  recovery_model.trainable_variables)
        
        # Apply the gradients to the Embedder and Recovery vars
        optimizer_embedder.apply_gradients(zip(gradients, # Always minimization function
                                           embedder_model.trainable_variables +
                                           recovery_model.trainable_variables))
        
        # Record the lower and upper layer gradients + the MSE for the autoencoder
        grad_embedder_ll(tf.norm(gradients[1]))
        grad_embedder_ul(tf.norm(gradients[3]))
        grad_recovery_ll(tf.norm(gradients[4]))
        grad_recovery_ul(tf.norm(gradients[6]))
        return R_loss_train
        
    @tf.function
    def distributed_train_step_embedder(X_train):
        per_replica_losses = mirrored_strategy.run(train_step_embedder, 
                                                   args=(X_train,))
        R_loss_train = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                                per_replica_losses, 
                                                axis=None)
        r_loss_train(R_loss_train / mirrored_strategy.num_replicas_in_sync)
    
    def test_step_embedder(X_test):
        # Apply the Embedder to data and Recovery to predicted hidden states
        e_pred_test = embedder_model(X_test, training=False)
        r_pred_test = recovery_model(e_pred_test, training=False)
        
        # Compute the loss function for the LSTM autoencoder using MirroredStrategy
        R_loss_test = compute_mse_loss(X_test, r_pred_test)
        return R_loss_test 
    
    @tf.function
    def distributed_test_step_embedder(X_test):
        per_replica_losses = mirrored_strategy.run(test_step_embedder, 
                                                   args=(X_test,))
        R_loss_test = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                               per_replica_losses, 
                                               axis=None)
        r_loss_test(R_loss_test / mirrored_strategy.num_replicas_in_sync)
    
    # Initialize the number of minibatches
    nr_mb_train = 0
    for epoch in range(load_epochs, load_epochs + 200):
        # Reset the metrics
        r_loss_train.reset_states()
        r_loss_test.reset_states()
        grad_embedder_ll.reset_states()
        grad_embedder_ul.reset_states()
        grad_recovery_ll.reset_states()
        grad_recovery_ul.reset_states()
       
        # Train over the complete train dataset
        for x_train in X_train:
            distributed_train_step_embedder(x_train)
            
            # Log the gradients in the bottom layer of embedder and recovery models
            with summary_writer_bottom.as_default():
                tf.summary.scalar('1. Pre-training Autoencoder/2. Gradient norm - embedder',
                                  grad_embedder_ll.result(), step=nr_mb_train,
                                  description = str(descr_pretrain_auto_embedder_grads()))
                tf.summary.scalar('1. Pre-training Autoencoder/2. Gradient norm - recovery',
                                  grad_recovery_ll.result(), step=nr_mb_train,
                                  description = str(descr_pretrain_auto_recovery_grads()))
            
            # Log the gradients in the top layer of embedder and recovery models
            with summary_writer_top.as_default():
                tf.summary.scalar('1. Pre-training Autoencoder/2. Gradient norm - embedder',
                                  grad_embedder_ul.result(), step=nr_mb_train)
                tf.summary.scalar('1. Pre-training Autoencoder/2. Gradient norm - recovery',
                                  grad_recovery_ul.result(), step=nr_mb_train)
            nr_mb_train += 1
        
        # Loop over the complete test dataset
        for x_test in X_test:
            distributed_test_step_embedder(x_test)
        
        # Log the recovery loss for the training dataset
        with summary_writer_train.as_default():
            tf.summary.scalar('1. Pre-training Autoencoder/1. Recovery loss', 
                              r_loss_train.result(), step=epoch,
                              description = str(descr_pretrain_auto_loss()))
            # Log histograms of the models weights per 50 epochs 
            if epoch % 50 == 0:
                add_hist(embedder_model.trainable_variables, epoch)
                add_hist(recovery_model.trainable_variables, epoch)
        
        # Log the recovery loss for the test dataset
        with summary_writer_test.as_default():
            tf.summary.scalar('1. Pre-training Autoencoder/1. Recovery loss', 
                              r_loss_test.result(), step=epoch)
       
        # Log the progress to the user console in Python    
        template = 'Autoencoder training: Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch+1, 
                              np.round(r_loss_train.result().numpy(),5),
                              np.round(r_loss_test.result().numpy(), 5)))
    print('Finished Embedding Network Training')

    # 2. Supervisor training 
    def train_step_supervised(X_train):
        with tf.GradientTape() as tape:
            # Compute embedding and supervised embedding on training dataset
            e_pred_train = embedder_model(X_train, training=True)
            H_hat_supervise = supervisor_model(e_pred_train, training=True)
      
            # Compute MSE for the Supervisor model on the training dataset
            G_loss_S_train = compute_mse_loss(e_pred_train[:, 1:, :],
                                              H_hat_supervise[:, 1:, :])
      
        # Compute the gradients w.r.t. the supervisor variables
        gradients = tape.gradient(G_loss_S_train, 
                                  supervisor_model.trainable_variables)
      
        # Apply the gradients to the supervisor variables
        optimizer_supervisor.apply_gradients(zip(gradients,
                                             supervisor_model.trainable_variables))
      
        # Record the lower and upper layer gradients
        grad_supervisor_ll(tf.norm(gradients[1]))
        grad_supervisor_ul(tf.norm(gradients[3]))

        # Return the train MSE for the supervisor model
        return G_loss_S_train
      
    @tf.function
    def distributed_train_step_supervised(X_train):
        per_replica_losses = mirrored_strategy.run(train_step_supervised,
                                                   args = (X_train,))
        G_loss_S_train = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                  per_replica_losses,
                                                  axis=None)
        g_loss_s_train(G_loss_S_train / mirrored_strategy.num_replicas_in_sync)
     
    def test_step_supervised(X_test):
        # Compute embedding and supervised embedding on test dataset
        e_pred_test = embedder_model(X_test, training=False)
        H_hat_supervise_test = supervisor_model(e_pred_test, training=False)
        
        # Compute MSE for the Supervisor model on the test dataset
        G_loss_S_test = compute_mse_loss(e_pred_test[:, 1:, :], 
                                         H_hat_supervise_test[:, 1:, :])
        
        # Return the test MSE for the supervisor model
        return G_loss_S_test
    
    @tf.function
    def distributed_test_step_supervised(X_test):
        per_replica_losses = mirrored_strategy.run(test_step_supervised,
                                                   args = (X_test,))
        G_loss_S_test = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                 per_replica_losses,
                                                 axis=None)
        g_loss_s_test(G_loss_S_test / mirrored_strategy.num_replicas_in_sync)
     
    # Initialize the number of minibatches
    nr_mb_train = 0
    for epoch in range(load_epochs, load_epochs + 100):
        # Reset the metrics
        g_loss_s_train.reset_states()
        g_loss_s_test.reset_states()
        grad_supervisor_ll.reset_states()
        grad_supervisor_ul.reset_states()
        
        # Train over the complete train dataset
        for x_train in X_train:
            distributed_train_step_supervised(x_train)

            # Log the gradients of the bottom layer in the supervisor model
            with summary_writer_bottom.as_default():
                tf.summary.scalar('2. Pre-training Supervisor/2. Gradient norm - supervisor',
                                  grad_supervisor_ll.result(), step=nr_mb_train)
            
            # Log the gradients of the top layer in the supervisor model
            with summary_writer_top.as_default():
                tf.summary.scalar('2. Pre-training Supervisor/2. Gradient norm - supervisor',
                                  grad_supervisor_ul.result(), step=nr_mb_train,
                                  description = str(descr_pretrain_supervisor_grads()))
            nr_mb_train += 1
           
        # Loop over the complete test dataset   
        for x_test in X_test:
            distributed_test_step_supervised(x_test)
        
        # Log the supervised loss for the training dataset
        with summary_writer_train.as_default():
            tf.summary.scalar('2. Pre-training Supervisor/1. Supervised loss', 
                              g_loss_s_train.result(), step=epoch)
            # Log histograms of the model weights per 50 epochs
            if epoch % 50 == 0:
                add_hist(supervisor_model.trainable_variables, epoch)
        
        # Log the supervised loss for the test dataset
        with summary_writer_test.as_default():
            tf.summary.scalar('2. Pre-training Supervisor/1. Supervised loss',
                              g_loss_s_test.result(), step=epoch,
                              description = str(descr_pretrain_supervisor_loss()))
        
        # Log the progress to the user console in Python
        template = 'Epoch {}, Train Loss: {}, Test loss: {}'
        print(template.format(epoch+1, 
                              np.round(g_loss_s_train.result().numpy(),8),
                              np.round(g_loss_s_test.result().numpy(),8) ) )
    print('Finished training with Supervised loss only')
    
    # 3. Jointly training all models in TimeGAN        
    @tf.function
    def distributed_feature_matching_loss(X_train, Z):
        per_replica_losses = mirrored_strategy.run(feature_matching_loss,
                                                   args=(X_train, Z))
        G_loss_V1 = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                             per_replica_losses,
                                             axis=None)
        g_loss_v1(G_loss_V1)
    
    def feature_matching_loss(X_train, Z):
        real = embedder_model(X_train, training=False)
        fake = generator_model(Z, training=False)

        g_loss_v1 = tf.abs((tf.nn.moments(real, [0])[0]) - (tf.nn.moments(fake, [0])[0]))
        g_loss_v2 = tf.abs(tf.sqrt(tf.nn.moments(real, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(fake, [0])[1] + 1e-6))
        g_loss_v = tf.add(g_loss_v1,  g_loss_v2)
        return tf.nn.compute_average_loss(g_loss_v, 
                                          global_batch_size=batch_size*mirrored_strategy.num_replicas_in_sync)
    
    @tf.function
    def distributed_train_step_jointly_generator(X_train,Z):
        per_replica_losses = mirrored_strategy.run(train_step_jointly_generator, 
                                                   args=(X_train,Z))
        G_loss_U_e = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                              per_replica_losses[0], 
                                              axis=None)
        G_loss_S = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                            per_replica_losses[1], 
                                            axis=None)
        G_loss_V = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_losses[2],
                                            axis=None)
        
        g_loss_u_e(G_loss_U_e / mirrored_strategy.num_replicas_in_sync)
        g_loss_s(G_loss_S / mirrored_strategy.num_replicas_in_sync)
        g_loss_v1(G_loss_V / mirrored_strategy.num_replicas_in_sync)

    def train_step_jointly_generator(X_train, Z, graphing=False, wasserstein=True):
      if graphing: # Only used for creating the graph
            with tf.GradientTape() as tape:
              # We need these steps to make the graph in Tensorboard complete
              dummy1 = embedder_model(X_train, training=True) # Real embedding
              dummy2 = generator_model(Z, training=True) # Fake embedding
              dummy4 = recovery_model(tf.concat([dummy1, dummy2], axis=0), training=True) # Recovery from embedding
              dummy3 = supervisor_model(tf.concat([dummy1, dummy2], axis=0), training=True) # Supervisor on embedding
              dummy5 = discriminator_model(tf.concat([dummy1, dummy2], axis=0), training=True) # Discriminator on embedding
      else:
        if wasserstein:
            with tf.GradientTape() as tape1:
                H = embedder_model(X_train, training=True)
                x_tilde = recovery_model(H, training=True)
          
                # Apply Generator to Z and apply Supervisor on fake embedding
                E_hat = generator_model(Z, training=True)
                H_hat = supervisor_model(E_hat, training=True)
        
                # The loss for the generator training only takes the last part of
                # the Wasserstein loss equation because the first part 
                # gives a gradient of 0 w.r.t. the generator network
                G_loss_U_e = compute_wasserstein_loss(E_hat, E_hat, 
                                                      generator=True, training=True)

                # 2. Generator - Supervised loss for fake embeddings
                G_loss_S = compute_mse_loss(E_hat[:, 1:, :], H_hat[:, 1:, :])
                
                # 3. Scan the feature matching loss such that we can track mode collapse
                G_loss_V = feature_matching_loss(X_train, Z)

                # Sum and multiply supervisor loss by eta for equal
                # contribution to generator loss function
                G_loss_1 = G_loss_U_e + eta * G_loss_S + kappa * G_loss_V 
            
            with tf.GradientTape() as tape2:
                H = embedder_model(X_train, training=True)
                x_tilde = recovery_model(H, training=True)

                #Apply Generator to Z and apply Supervisor on fake embedding
                E_hat = generator_model(Z, training=True)
                H_hat = supervisor_model(E_hat, training=True)

                # Generator loss
                G_loss_U_e = compute_wasserstein_loss(E_hat, E_hat,
                                                      generator=True, training=True)

                # 2. Generator - Supervised loss for fake embeddings
                G_loss_S = compute_mse_loss(E_hat[:, 1:, :], H_hat[:, 1:, :])
                
                # 3. Scan the feature matching loss such that we can track mode collapse
                G_loss_V = feature_matching_loss(X_train, Z)

                # Sum and multiply supervisor loss by eta for equal
                # contribution to generator loss function
                G_loss_2 = G_loss_U_e + eta * G_loss_S + kappa * G_loss_V
            
            # Compute the gradients w.r.t.the generator model
            gradients_generator=tape1.gradient(G_loss_1,
                                               generator_model.trainable_variables)

            # Apply the gradients to the generator model
            optimizer_generator.apply_gradients(zip(gradients_generator, 
                                                generator_model.trainable_variables))
            
            # Compute the gradients w.r.t. the supervisor model
            gradients_supervisor=tape2.gradient(G_loss_2,
                                                supervisor_model.trainable_variables)

            # Apply the gradients to the supervisor model
            optimizer_supervisor.apply_gradients(zip(gradients_supervisor,
                                                 supervisor_model.trainable_variables))

            grad_generator_ll(tf.norm(gradients_generator[1]))
            grad_generator_ul(tf.norm(gradients_generator[9]))
        else:
            with tf.GradientTape() as tape1:
                H = embedder_model(X_train, training=True)
            
                # Apply Generator to Z and apply Supervisor on fake embedding
                E_hat = generator_model(Z, training=True)
                H_hat = supervisor_model(E_hat, training=True)
      
                # Compute real and fake probabilities using Discriminator model
                # 1. Generator - Adversarial loss - Minimize -log(D(G(z))) (saturated gradient approach)
                G_loss_U_e = compute_adversarial_loss(tf.ones_like(E_hat), 
                                                      E_hat, generator=True)
            
                # 2. Generator - Supervised loss for fake embeddings
                G_loss_S = compute_mse_loss(E_hat[:, 1:, :], H_hat[:, 1:, :])
                
                # 3. Generator - Feature Matching loss
                G_loss_V = feature_matching_loss(X_train, Z)

                # Sum and multiply supervisor loss by eta for equal
                # contribution to generator loss function
                G_loss_1 = G_loss_U_e + eta * G_loss_S + kappa * G_loss_V

            with tf.GradientTape() as tape2:
                H = embedder_model(X_train, training=True)

                # Apply Generator to Z and apply Supervisor on fake embedding
                E_hat = generator_model(Z, training=True)
                H_hat = supervisor_model(H, training=True)

                # Compute real and fake probabilities using Discriminator model
                # 1. Generator - Adversial loss - Minimize -log(D(G(z))) (saturated gradient approach)
                G_loss_U_e = compute_adversarial_loss(tf.ones_like(E_hat),
                                                      E_hat, generator=True)
                
                # 2. Generator - Supervised loss for fake embeddings
                G_loss_S = compute_mse_loss(E_hat[:, 1:, :], H_hat[:, 1:, :])

                # 3. Generator - Feature Matching loss
                G_loss_V = feature_matching_loss(X_train, Z)
            
                # Sum and multiply supervisor loss by eta for equal
                # contribution to generator loss function
                G_loss_2 = G_loss_U_e + eta * G_loss_S + kappa * G_loss_V
                          
            # Compute the gradients w.r.t. generator
            gradients_generator=tape1.gradient(G_loss_1,
                                               generator_model.trainable_variables)
          
            # Apply the gradients to the generator model
            optimizer_generator.apply_gradients(zip(gradients_generator, 
                                                generator_model.trainable_variables)) 

            # Compute the gradients w.r.t. supervisor model
            gradients_supervisor=tape2.gradient(G_loss_2,
                                                supervisor_model.trainable_variables)

            # Apply the gradients to the supervisor model
            optimizer_supervisor.apply_gradients(zip(gradients_supervisor,
                                                 supervisor_model.trainable_variables))

            # Record the lower and upper layer gradients + the MSE for the generator
            grad_generator_ll(tf.norm(gradients_generator[1]))
            grad_generator_ul(tf.norm(gradients_generator[9]))
        
      return [G_loss_U_e, G_loss_S, G_loss_V]
    
    @tf.function
    def distributed_test_step_jointly_generator(Z):
        per_replica_losses = mirrored_strategy.run(test_step_jointly_generator,
                                                   args=(Z,))
        G_loss_U_e_test = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                   per_replica_losses[0],
                                                   axis=None)
        G_loss_S_test = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                 per_replica_losses[1],
                                                 axis=None)

        g_loss_u_e_test(G_loss_U_e_test / mirrored_strategy.num_replicas_in_sync)
        g_loss_s_test(G_loss_S_test / mirrored_strategy.num_replicas_in_sync) 

    def test_step_jointly_generator(Z, wasserstein=True):
        E_hat = generator_model(Z, training=False)
        H_hat = supervisor_model(E_hat, training=False)
        
        if wasserstein:
            G_loss_U_e_test = compute_wasserstein_loss(E_hat, E_hat, 
                                                       generator=True, training=False)
        else:
            G_loss_U_e_test = compute_adversarial_loss(tf.ones_like(E_hat), E_hat, visualization=True)
        
        G_loss_S_test = compute_mse_loss(E_hat[:, 1:, :], H_hat[:, 1:, :])
        return [G_loss_U_e_test, G_loss_S_test]
        
    @tf.function
    def distributed_train_step_jointly_embedder(X_train):
        per_replica_losses = mirrored_strategy.run(train_step_jointly_embedder, 
                                                   args=(X_train,))
        r_loss_train = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                                per_replica_losses[0], 
                                                axis=None)
        G_loss_S_embedder = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                                     per_replica_losses[1], 
                                                     axis=None)
        e_loss_T0(r_loss_train / mirrored_strategy.num_replicas_in_sync)
        g_loss_s_embedder(G_loss_S_embedder / mirrored_strategy.num_replicas_in_sync)
                
    def train_step_jointly_embedder(X_train):
        with tf.GradientTape() as tape1:
            # Apply Embedder to data and recover the data from the embedding space
            H = embedder_model(X_train, training=True) 
            X_tilde = recovery_model(H, training=True)
          
            # Compute the loss function for the embedder-recovery model
            r_loss_train = compute_mse_loss(X_train, X_tilde)  
          
            # Include the supervision loss but only for 10 %
            H_hat_supervise = supervisor_model(H, training=True)
            G_loss_S_embedder = compute_mse_loss(H[:,1:,:], H_hat_supervise[:,1:,:])
          
            # Combine the two losses
            E_loss1 = r_loss_train + lambda_val * G_loss_S_embedder
        
        with tf.GradientTape() as tape2:
            # Apply Embedder to data and recover the data from the embedding space
            H = embedder_model(X_train, training=True)
            X_tilde = recovery_model(H, training=True)

            # Compute the loss function for the embedder-recovery model
            r_loss_train = compute_mse_loss(X_train, X_tilde)

            # Include the supervision loss
            H_hat_supervise = supervisor_model(H, training=True)
            G_loss_S_embedder = compute_mse_loss(H[:, 1:,:], H_hat_supervise[:, 1:,:])

            # Combine the two losses
            E_loss2 = r_loss_train + lambda_val * G_loss_S_embedder
        
        # Compute the gradients w.r.t. the embedder-recovery network
        gradients_embedder=tape1.gradient(E_loss1,
                                          embedder_model.trainable_variables +
                                          recovery_model.trainable_variables)
        
        # Compute the gradients w.r.t. the supervisor network
        gradients_supervisor=tape2.gradient(E_loss2,
                                            supervisor_model.trainable_variables)
        
        # Apply the gradients to the embedder-recovery network
        optimizer_embedder.apply_gradients(zip(gradients_embedder,
                                           embedder_model.trainable_variables + 
                                           recovery_model.trainable_variables))
        
        # Apply the gradients to the supervisor network
        optimizer_supervisor.apply_gradients(zip(gradients_supervisor,
                                             supervisor_model.trainable_variables))

        return [r_loss_train, G_loss_S_embedder]
    
    @tf.function
    def distributed_test_step_jointly_embedder(X_test):
        per_replica_losses = mirrored_strategy.run(test_step_jointly_embedder, 
                                                   args=(X_test,))
        E_loss_T0_test = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                                  per_replica_losses[0], 
                                                  axis=None)
        G_loss_S_embedder_test = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                                          per_replica_losses[1], 
                                                          axis=None)
        e_loss_T0_test(E_loss_T0_test / mirrored_strategy.num_replicas_in_sync)
        g_loss_s_embedder_test(G_loss_S_embedder_test / mirrored_strategy.num_replicas_in_sync)
        
    def test_step_jointly_embedder(X_test):
        # Compute embedding for the test dataset
        H = embedder_model(X_test, training=False)
        
        # Compute recovery for the embedding
        X_tilde = recovery_model(H, training=False)
        
        # Compute MSE for the Autoencoder network
        E_loss_T0_test = compute_mse_loss(X_test, X_tilde)

        # Compute supervised embedding
        H_hat_supervise = supervisor_model(H, training=False)

        # Compute MSE for supervisor model
        G_loss_S_embedder_test = compute_mse_loss(H[:,1:,:], H_hat_supervise[:,1:,:])
        
        return [E_loss_T0_test, G_loss_S_embedder_test]
    
    @tf.function
    def distributed_train_step_discriminator(X_train, Z):
        """Perform the train step for the Discriminator model using the
        MirroredStrategy approach."""
        # Run train step for Discriminator using MirroredStrategy
        per_replica_losses = mirrored_strategy.run(train_step_discriminator, 
                                                   args=(X_train, Z))

        # Sum individual Discriminator losses over GPUs for train dataset
        D_loss_train = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                                per_replica_losses[0], 
                                                axis=None)

        # Sum individual Discriminator losses over GPUs for real samples
        D_loss_real = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                               per_replica_losses[1],
                                               axis = None)

        # Sum individual Discriminator losses over GPUs for fake samples
        D_loss_fake = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                               per_replica_losses[2],
                                               axis=None)

        # Log Discriminator training data losses and divide by number of GPUs
        d_loss(D_loss_train / mirrored_strategy.num_replicas_in_sync)
        d_loss_real(D_loss_real / mirrored_strategy.num_replicas_in_sync)
        d_loss_fake(D_loss_fake / mirrored_strategy.num_replicas_in_sync)
    
    def train_step_discriminator(X_train, Z, smoothing_factor=0.9, wasserstein=True):
        if wasserstein: # Wasserstein Gradient Penalty (Gulrajani et al., 2017)
            with tf.GradientTape() as tape:
                # Embeddings for real data and classifications from discriminator
                H = embedder_model(X_train, training=True)
                # Embeddings for fake data and classifications from discriminator
                E_hat = generator_model(Z, training=True)
                
                # Compute the gradient penalty on a convex combination of real and fake
                alpha = np.random.uniform(size=(H.shape[0], 20, hidden_dim))
                alpha = tf.convert_to_tensor(alpha, dtype=tf.float16)
                interpolates = H + alpha * (E_hat - H)

                # Compute the critic values
                #critic_val = discriminator_model(interpolates, training=True)

                # Compute the gradients, the norms and the gradient penalty
                # We added a epsilon (1e-8) such that the norms does not give Inf or NaN
                gradients = K.gradients(discriminator_model(interpolates, training=True), interpolates) # (32, 20, 4)
                # We try another norm calculation
                #norms = tf.square(tf.norm(gradients[0], ord=2) -1.0)
                
                #print(norms)
                norms = tf.sqrt(1e-4 + tf.reduce_sum(tf.square(gradients), axis=[2,3])) # (1, 32)
                gp = tf.reduce_mean((norms - 1)**2) # (, )
                #print(gp)
                
                if tf.math.is_nan(gp):
                    gp=tf.constant(0.0, dtype=tf.float16)
                    print('gp is a NaN value.')

                # Record the gradient penalty for the training dataset
                w_loss_gp(gp)

                # Compute Wasserstein loss estimate (Arjovsky et al., 2017)
                D_loss = compute_wasserstein_loss(H, E_hat, 
                                                  generator=False, training=True) # (,)
                
                # Define Wasserstein loss with gradient penalty
                WGAN_GP = D_loss + gamma * gp

            D_loss_real = - compute_wasserstein_loss(H, H, generator=True, 
                                                     training=True)
            D_loss_fake_e = - compute_wasserstein_loss(E_hat, E_hat, generator=True,
                                                       training=True)
            # Compute gradients w.r.t. the discriminator model
            grad_d=tape.gradient(WGAN_GP,
                                 discriminator_model.trainable_variables)
                
            # Apply the gradient to the discriminator model
            optimizer_discriminator.apply_gradients(zip(grad_d,
                                                    discriminator_model.trainable_variables))
            # Record the lower and upper layer gradients for the discriminator model
            grad_discriminator_ll(tf.norm(grad_d[1]))
            grad_discriminator_ul(tf.norm(grad_d[6]))
        else: # Compute Kullback-Leibler divergence (Goodfellow et al, 2014)
            with tf.GradientTape() as tape:
                # Embeddings for real data and classifications from discriminator
                H = embedder_model(X_train, training=True)
                # Embeddings for fake data and classifications from discriminator
                E_hat = generator_model(Z, training=True)

                # Compute the classification for real and fake samples
                Y_real = discriminator_model(H, training=True)
                Y_fake_e = discriminator_model(E_hat, training=True)

                # Compute the negative cross entropy for real and fake samples
                D_loss_real = compute_adversarial_loss(tf.ones_like(Y_real) * smoothing_factor, Y_real)
                D_loss_fake_e = compute_adversarial_loss(tf.zeros_like(Y_fake_e), 
                                                         Y_fake_e)
                D_loss = D_loss_real + D_loss_fake_e
            
            # Compute the gradients w.r.t. the discriminator model
            grad_d=tape.gradient(D_loss,
                                 discriminator_model.trainable_variables)
                
            # Apply the gradient to the discriminator model
            optimizer_discriminator.apply_gradients(zip(grad_d,
                                                    discriminator_model.trainable_variables))
            
            # Record the lower and upper layer gradients for the discriminator model
            grad_discriminator_ll(tf.norm(grad_d[1]))
            grad_discriminator_ul(tf.norm(grad_d[6]))
        return [WGAN_GP, D_loss_real, D_loss_fake_e]
    
    @tf.function
    def distributed_test_step_discriminator(X_test, Z):
        """Perform the test step for the Discriminator model using the 
        MirroredStrategy and log the Discriminator loss for the test dataset."""
        # Run test step for Discriminator using MirroredStrategy 
        per_replica_losses = mirrored_strategy.run(test_step_discriminator, 
                                                   args=(X_test, Z))

        # Sum individual Discriminator losses over GPUs for test dataset
        D_loss_test = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                               per_replica_losses, 
                                               axis=None)

        # Log Discriminator loss for test data and divide by number of GPUs
        d_loss_test(D_loss_test / mirrored_strategy.num_replicas_in_sync)
            
    def test_step_discriminator(X_test, Z, smoothing_factor=0.9, wasserstein=True):
        # Compute embedding for the test dataset
        H = embedder_model(X_test, training=False)

        # Generate embedding
        E_hat = generator_model(Z, training=False)
        if wasserstein:
            # Compute a convex combination for the gradient penalty
            alpha = np.random.uniform(size=(H.shape[0], 20, hidden_dim))
            alpha = tf.convert_to_tensor(alpha, dtype=tf.float16)
            interpolates = H + alpha * (E_hat - H)

            # Compute the critic values
            # critic_val = discriminator_model(interpolates, training=False)

            # Compute the gradients, the norms, and the gradient penalty
            # Added an epsilon such that the gradient penalty does not explode
            gradients = K.gradients(discriminator_model(interpolates, training=False), interpolates) # (32, 20, 4)
            norms = tf.sqrt(1.0e-4 + tf.reduce_sum(tf.square(gradients), axis=[2,3]))
            
            gp_test = tf.reduce_mean((norms- 1)**2) # (,)
            
            if tf.math.is_nan(gp_test):
                gp_test = tf.constant(0.0, dtype=tf.float16)
                print('gp_test is a NaN value.')
            # Record the gradient penalty
            w_loss_gp_test(gp_test)

            # Compute Wasserstein loss estimate (Arjovsky et al., 2017)
            D_loss_test = compute_wasserstein_loss(H, E_hat, 
                                                   generator=False, training=False)
            D_loss_real = - compute_wasserstein_loss(H, H, generator=True,
                                                     training=False)
            D_loss_fake_e = - compute_wasserstein_loss(E_hat, E_hat, generator=True,
                                                       training=False)

        else:
            # Compute classifications for real and fake samples
            Y_real = discriminator_model(H, training=False) 
            Y_fake_e = discriminator_model(E_hat, training=False)

            # Compute negative cross entropy loss for real and fake samples
            D_loss_real = compute_adversarial_loss(tf.ones_like(Y_real) * smoothing_factor, Y_real)
            D_loss_fake_e = compute_adversarial_loss(tf.zeros_like(Y_fake_e), 
                                                     Y_fake_e)
            D_loss_test = D_loss_real + D_loss_fake_e
        return D_loss_test
    
    @tf.function
    def distributed_evaluate_accuracy(X_test, Z):
        """Returns the accuracy for a test dataset and a fakely generated dataset.
           Only use accuracy measure when you implement negative cross entropy
           loss as the loss function for the Discriminator in a Generative 
           Adversarial Network."""
        # Run the accuracy evaluation step on the test dataset
        per_replica_losses = mirrored_strategy.run(evaluate_accuracy,
                                                   args=(X_test, Z))

        # Sum the individual accuracies over GPUs for real test samples
        D_accuracy_real = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                   per_replica_losses[0],
                                                   axis=None)

        # Sum individual accuracies over GPUs for fake test samples
        D_accuracy_fake = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                   per_replica_losses[1],
                                                   axis=None)
        # Log accuracies for real and fake datasets and divide by numer of GPUs
        return [D_accuracy_real / mirrored_strategy.num_replicas_in_sync, D_accuracy_fake / mirrored_strategy.num_replicas_in_sync]

    def evaluate_accuracy(X_test, Z):
        # Compute whether the Discriminator prediction is greater than 0.5
        Y_real_test = tf.cast(tf.math.greater(discriminator_model.predict(embedder_model(X_test, training=False)), tf.constant(0.5, dtype=tf.float16)), tf.float16)
        
        # Compute whether the Discriminator prediction is less than 0.5
        Y_fake_test = tf.cast(tf.math.less(discriminator_model.predict(generator_model(Z, training=False)), tf.constant(0.5, dtype=tf.float16)), tf.float16)
        
        # Count the number of good classifications for the real dataset
        number_good = tf.reduce_sum(tf.cast(tf.equal(Y_real_test, tf.constant(1, dtype=tf.float16)), tf.float16))
        # Count the number of good classification for the fake dataset
        number_false = tf.reduce_sum(tf.cast(tf.equal(Y_fake_test, tf.constant(1, dtype=tf.float16)), tf.float16))
        # Compute the accuracy for the real and fake datasets
        D_accuracy_real = tf.divide(number_good, Y_real_test.shape[0]* Y_real_test.shape[1])
        D_accuracy_fake = tf.divide(number_false, Y_fake_test.shape[0] * Y_fake_test.shape[1])
        return [D_accuracy_real, D_accuracy_fake]
    
    # Helper counter for the already performed epochs
    already_done_epochs = epoch
        
    # Define the algorithm for training jointly
    print('Start joint training')
    nr_mb_train = 0 # Iterator for generator training
    o = 0 # Iterator for discriminator train iterations
    e = 0 # Iterator for discriminator test iterations
    for epoch in range(load_epochs, iterations+load_epochs):
        g_loss_u_e.reset_states() # Reset the loss at every epoch
        g_loss_s.reset_states()
        e_loss_T0.reset_states()
        g_loss_s_embedder.reset_states()
        d_loss.reset_states()
        d_loss_real.reset_states()
        d_loss_fake.reset_states()
        g_loss_v1.reset_states()
        w_loss_gp.reset_states()
        w_loss_gp_test.reset_states()
        
        g_loss_u_e_test.reset_states()
        d_loss_test.reset_states()
        
        # This loop is GENERATOR TRAINING
        r = 0       
        for x_train in X_train:          
            if r < 19:
                # Sample the random numbers to use in minibatch training
                Z_mb = RandomGenerator(int(batch_size / mirrored_strategy.num_replicas_in_sync), [20, hidden_dim])
                # Train the Generator and Supervisor network jointly
                distributed_train_step_jointly_generator(x_train, Z_mb)

                # Train the Autoencoder and Supervisor network jointly
                distributed_train_step_jointly_embedder(x_train)
                
                with summary_writer_bottom.as_default():
                    tf.summary.scalar('3. TimeGAN training - GAN/3. Gradient norm - generator',
                                      grad_generator_ll.result(), step=nr_mb_train)
                with summary_writer_top.as_default():                              
                    tf.summary.scalar('3. TimeGAN training - GAN/3. Gradient norm - generator',
                                      grad_generator_ul.result(), step=nr_mb_train,
                                      description = str(descr_joint_generator_grads()))
                nr_mb_train += 1
            r += 1
        
        # Write the Autoencoder and Supervised loss for the training data to a Tensorboard
        with summary_writer_train.as_default():
            tf.summary.scalar('3. TimeGAN training - Autoencoder/1. Recovery loss',
                             e_loss_T0.result(), step=epoch,
                             description = str(descr_joint_auto_loss()))
            tf.summary.scalar('3. TimeGAN training - Autoencoder/1. Supervised loss',
                             g_loss_s_embedder.result(), step=epoch,
                             description = str(descr_joint_supervisor_loss()))
        
        r = 0    
        for x_test in X_test:
            if r < 9:
                # Sample the random numbers to use in minibatch training
                Z_mb = RandomGenerator(int(batch_size / mirrored_strategy.num_replicas_in_sync), [20, hidden_dim])
                # Evaluate Generator and Supervisor jointly
                distributed_test_step_jointly_generator(Z_mb)

                # Evaluate Autoencoder and Supervisor jointly
                distributed_test_step_jointly_embedder(x_test)
                
            r += 1

        # Write the Autoencoder and Supervised loss for the test data to a Tensorboard
        with summary_writer_test.as_default(): 
            tf.summary.scalar('3. TimeGAN training - Autoencoder/1. Recovery loss',
                              e_loss_T0_test.result(), step=epoch)
            tf.summary.scalar('3. TimeGAN training - Autoencoder/1. Supervised loss',
                              g_loss_s_embedder_test.result(), step=epoch)
   
        # This loop is DISCRIMINATOR TRAINING
        i = 0 # Hyperparameter for the number of discriminator iterations
        while i < 5:
            q = 0
            for x_train in X_train: 
                if q < 19:
                    # Sample the random numbers to use in minibatch training 
                    Z_mb = RandomGenerator(int(batch_size / mirrored_strategy.num_replicas_in_sync), [20, hidden_dim])
                    distributed_train_step_discriminator(x_train, Z_mb)

                    with summary_writer_gp.as_default():
                        # Log the sample mean gradient penalty for every train minibatch
                        tf.summary.scalar('3. TimeGAN training - GAN/4. Gradient penalty',
                                          w_loss_gp.result(), step=o,
                                          description = str(descr_joint_gradient_penalty()))

                    with summary_writer_top.as_default():
                        # Log gradient norm for the upper layer
                        tf.summary.scalar('3. TimeGAN training - GAN/3. Gradient norm - discriminator',
                                          grad_discriminator_ul.result(), step=o,
                                          description = str(descr_joint_discriminator_grads()))
                    with summary_writer_bottom.as_default():
                        # Log gradient norm for the lower layer
                        tf.summary.scalar('3. TimeGAN training - GAN/3. Gradient norm - discriminator',
                                          grad_discriminator_ll.result(), step=o)
                    o += 1
                q +=1
            
            q = 0 # Per Replica losses does not work for the last batch
            for x_test in X_test:
                if q < 9:
                    Z_mb = RandomGenerator(int(batch_size / mirrored_strategy.num_replicas_in_sync), [20, hidden_dim])
                    distributed_test_step_discriminator(x_test, Z_mb)

                    with summary_writer_gp_test.as_default():
                        # Log the sample mean gradient penalty for every test minibatch
                        tf.summary.scalar('3. TimeGAN training - GAN/4. Gradient penalty',
                                          w_loss_gp_test.result(), step=e)
                    e += 1
                q += 1
            i += 1 # Iterator for repetitive discriminator trainings
        
        # Only use with WGAN-GP
        # Check whether the gradient penalty is not exploding with a 2-layer LSTM
        if w_loss_gp.result() > 5:
            for _ in range(5): # 5 extra discriminator training iterations such that gradient doesnt explode 
                q = 0 # Iterator
                for x_train in X_train:
                    if q < 19:
                        Z_mb = RandomGenerator(int(batch_size / mirrored_strategy.num_replicas_in_sync), [20, hidden_dim])
                        distributed_train_step_discriminator(x_train, Z_mb)  
                    q += 1   

        # Compute the test accuracy - Not when we use Wasserstein distance
        # beceause the discriminator then becomes a critic
        #acc_real_array = np.array([])
        #acc_fake_array = np.array([])
        #for x_test in X_test:
        #    Z_mb = RandomGenerator(int(batch_size / mirrored_strategy.num_replicas_in_sync), [20, hidden_dim])
        #    acc_real, acc_fake = distributed_evaluate_accuracy(x_test, Z_mb)
        #    acc_real_array = np.append(acc_real_array, acc_real)
        #    acc_fake_array = np.append(acc_fake_array, acc_fake)
        
        # Write all train losses to a TF 2.0 Tensorboard
        with summary_writer_train.as_default():
            tf.summary.scalar('3. TimeGAN training - GAN/1. Unsupervised loss - Discriminator',
                              d_loss.result(), step=epoch,
                              description = str(descr_joint_discriminator_loss_wasserstein()))
            tf.summary.scalar('3. TimeGAN training - GAN/1. Unsupervised loss - Generator',
                              g_loss_u_e.result(), step=epoch,
                              description = str(descr_joint_generator_loss_wasserstein()))
            tf.summary.scalar('3. TimeGAN training - GAN/1. Supervised loss',
                              g_loss_s.result(), step=epoch,
                              description = str(descr_joint_supervisor_loss()))
            tf.summary.scalar('3. TimeGAN training - GAN/1. Feature Matching loss',
                              g_loss_v1.result(), step=epoch,
                              description = str(descr_joint_feature_matching_loss()))
        
        # Write all test losses to a TF 2.0 Tensorboard
        with summary_writer_test.as_default():
            tf.summary.scalar('3. TimeGAN training - GAN/1. Unsupervised loss - Discriminator',
                              d_loss_test.result(), step=epoch)
            tf.summary.scalar('3. TimeGAN training - GAN/1. Unsupervised loss - Generator',
                              g_loss_u_e_test.result(), step=epoch)
            tf.summary.scalar('3. TimeGAN training - GAN/1. Supervised loss',
                              g_loss_s_test.result(), step=epoch)
        
        with summary_writer_real_data.as_default():  
            #tf.summary.scalar('3. TimeGAN training - GAN/2. Accuracy',
            #                  tf.reduce_mean(acc_real_array), step=epoch)
            tf.summary.scalar('3. TimeGAN training - GAN/1. Unsupervised loss - Discriminator',
                              d_loss_real.result(), step=epoch)

        with summary_writer_fake_data.as_default():
            #tf.summary.scalar('3. TimeGAN training - GAN/2. Accuracy',
            #                  tf.reduce_mean(acc_fake_array), step = epoch,
            #                  description = str(descr_joint_accuracy()))
            
            tf.summary.scalar('3. TimeGAN training - GAN/1. Unsupervised loss - Discriminator',
                              d_loss_fake.result(), step=epoch)
        
        # Log the model weights in histograms in a TF 2.0 Tensorboard
        with summary_writer_train.as_default():
            if epoch % 50 == 0: # First pre-trained models
                add_hist(embedder_model.trainable_variables,
                         epoch + already_done_epochs) 
                add_hist(recovery_model.trainable_variables,
                         epoch + already_done_epochs) 
                add_hist(supervisor_model.trainable_variables,
                         epoch + already_done_epochs)
                add_hist(generator_model.trainable_variables, epoch) # Then not pre-trained models
                add_hist(discriminator_model.trainable_variables, epoch)
            
            # Save all weights for every 150 epochs
            if epoch % 50 == 0 and epoch != 0:
                embedder_model.save_weights(log_dir+'/weights/embedder/epoch_' + str(epoch))
                recovery_model.save_weights(log_dir+'/weights/recovery/epoch_' + str(epoch))
                supervisor_model.save_weights(log_dir+'/weights/supervisor/epoch_' + str(epoch))
                generator_model.save_weights(log_dir+'/weights/generator/epoch_' + str(epoch))
                discriminator_model.save_weights(log_dir+'/weights/discriminator/epoch_' + str(epoch)) 
            
            # Print the progress to the Python console
            print('step: '+ str(epoch+1) +  
                  ', g_loss_u_e: ' + str(np.round(g_loss_u_e.result().numpy(),8)) + 
                  ', g_loss_s: ' + str(np.round(g_loss_s.result().numpy(),8)) +
                  ', g_loss_v: ' + str(np.round(g_loss_v1.result().numpy(),8)) +
                  ', g_loss_s_embedder: ' + 
                  str(np.round(g_loss_s_embedder.result().numpy(),8)) +
                  ', e_loss_t0: ' + str(np.round(e_loss_T0.result().numpy(),8)) + 
                  ', d_loss: ' + str(np.round(d_loss.result().numpy(),8)) +
                  ', d_loss_real: ' + str(np.round(d_loss_real.result().numpy(), 8)) +
                  ', d_loss_fake: ' + str(np.round(d_loss_fake.result().numpy(), 8)) + 
                  ', w_loss_gp: ' + str(np.round(w_loss_gp.result().numpy(), 8)) +
                  ', w_loss_gp_test: ' + str(np.round(w_loss_gp_test.result().numpy(), 8))) 
    print('Finish joint training')
