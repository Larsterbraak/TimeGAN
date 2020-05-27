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

def run(parameters, hparams):
    
    # Network Parameters
    hidden_dim   = parameters['hidden_dim']  #
    num_layers   = parameters['num_layers']  # Still have to implement
    iterations   = parameters['iterations']  # Test run to check for overfitting
    batch_size   = parameters['batch_size']  # Currently locked at 25
    module_name  = parameters['module_name'] # 'lstm' or 'GRU''
    z_dim        = parameters['z_dim']       # Currently locked at 5
    gamma        = 1
    
    # Define the TensorBoard such that we can visualize the results
    log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Create an instance of all neural networks models (All LSTM)
    embedder_model = Embedder('logs/embedder', hparams)
    recovery_model = Recovery('logs/recovery', hparams)
    generator_model = Generator('logs/generator', hparams)
    supervisor_model = Supervisor('logs/supervisor', hparams)
    discriminator_model = Discriminator('logs/TimeGAN', hparams)
    
    # Metrics to track during training
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    g_loss_s_train = tf.keras.metrics.Mean(name='g_loss_s_train')
    g_loss_s_test = tf.keras.metrics.Mean(name='g_loss_s_test')
    d_loss = tf.keras.metrics.Mean(name='d_loss')
    g_loss_u = tf.keras.metrics.Mean(name='g_loss_u')
    g_loss_s = tf.keras.metrics.Mean(name='g_loss_s')
    g_loss_v = tf.keras.metrics.Mean(name='g_loss_v')
    e_loss_T0 = tf.keras.metrics.Mean(name='e_loss_T0')
    
    # Create the loss object, optimizer, and training function
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(0.001)
    
    # Create loss object for sigmoid (binary) cross entropy
    loss_object_adversarial = tf.losses.BinaryCrossentropy(from_logits=True)
    # from_logits = True because the last dense layers is linear and
    # does not have an activation -- could be differently specified
    
    # 1. Start with embedder training (Optimal LSTM auto encoder network)
    @tf.function
    def train_step_embedder(X_train):
        with tf.GradientTape() as tape:
            # Apply Embedder to data and Recovery to predicted hidden states 
            e_pred_train = embedder_model(X_train)
            r_pred_train = recovery_model(e_pred_train)
            
            # Compute loss for LSTM autoencoder and check if non-negative
            r_loss_train = loss_object(X_train, r_pred_train)
            tf.debugging.assert_non_negative(r_loss_train)
        
        # Compute the gradients with respect to the Embedder and Recovery vars
        gradients = tape.gradient(r_loss_train, 
                                  embedder_model.trainable_variables +
                                  recovery_model.trainable_variables)
        
        # Apply the gradients to the Embedder and Recovery vars
        optimizer.apply_gradients(zip(gradients, 
                                      embedder_model.trainable_variables +
                                      recovery_model.trainable_variables))
        train_loss(r_loss_train)
      
    @tf.function
    def test_step_embedder(X_test):
        # Apply the Embedder to data and Recovery to predicted hidden states
        e_pred_test = embedder_model(X_test)
        r_pred_test = recovery_model(e_pred_test)
        
        # Compute the loss function for the LSTM autoencoder
        r_loss_test = loss_object(X_test, r_pred_test)
        test_loss(r_loss_test)    
    
    # Train the embedder for the input data
    for epoch in range(iterations):
        train_loss.reset_states()
        test_loss.reset_states()
        
        # Train over the complete train and test dataset
        for x_train in X_train:
            train_step_embedder(x_train)
        
        for x_test in X_test:
            test_step_embedder(x_test)
        
        with summary_writer.as_default():
            tf.summary.scalar('recovery/train', train_loss.result(),step=epoch)
            tf.summary.scalar('recovery/test', test_loss.result(), step=epoch)
            add_hist(embedder_model.trainable_variables, epoch)
            add_hist(recovery_model.trainable_variables, epoch)
        
        # Log the progress to the user console in python    
        template = 'training: Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch+1, train_loss.result(),test_loss.result()))
    
    print('Finished Embedding Network Training')

    # 2. Continue with supervised loss (Optimal temporal relations over time)
    @tf.function
    def train_step_supervised(X_train):
      with tf.GradientTape() as tape:
        # Apply Embedder to data and check temporal relations with supervisor
        e_pred_train = embedder_model(X_train)
        H_hat_supervise = supervisor_model(e_pred_train)
        
        # Compute squared loss for real embedding and supervised embedding
        G_loss_S = loss_object(e_pred_train[:, 1:, :],
                               H_hat_supervise[:, 1:, :])
      
      # Compute the gradients with respect to the Embedder and Recovery vars
      gradients = tape.gradient(G_loss_S, 
                                supervisor_model.trainable_variables)
      
      # Apply the gradients to the Embedder and Recovery vars
      optimizer.apply_gradients(zip(gradients, 
                                    supervisor_model.trainable_variables))
      
      # Compute the training loss for the supervised model
      g_loss_s_train(G_loss_S)
      
    @tf.function
    def test_step_supervised(X_test):
        e_pred_test = embedder_model(X_test)
        H_hat_supervise_test = supervisor_model(e_pred_test)
        G_loss_S = loss_object(e_pred_test[:, 1:, :], 
                               H_hat_supervise_test[:, 1:, :])
        g_loss_s_test(G_loss_S)
    
    for epoch in range(iterations):
        g_loss_s_train.reset_states()
        g_loss_s_test.reset_states()
        
        for x_train in X_train:
            train_step_supervised(x_train)
        
        for x_test in X_test:
            test_step_supervised(x_test)
        
        with summary_writer.as_default():
            tf.summary.scalar('supervisor/train', g_loss_s_train.result(),
                              step=epoch)
            tf.summary.scalar('supervisor/test', g_loss_s_test.result(),
                              step=epoch)
            add_hist(generator_model.trainable_variables, epoch)
            add_hist(supervisor_model.trainable_variables, epoch)
                
        template = 'Epoch {}, Loss: {}, Test loss: {}'
        print(template.format(epoch+1, g_loss_s_train.result(),
                              g_loss_s_test.result() ) )
    print('Finished training with Supervised loss only')
    
    # 3. Continue with joint training
    @tf.function
    def train_step_jointly_generator(X_train, Z):
        with tf.GradientTape() as tape:
          # Apply Embedder to data and apply Supervisor 
          H = embedder_model(X_train) 
          H_hat_supervise = supervisor_model(H)
          
          # Apply Generator to Wiener process andd apply Supervisor
          E_hat = generator_model(Z)
          H_hat = supervisor_model(E_hat)
          
          # Create synthetic data from fakely generated embedding space
          X_hat = recovery_model(H_hat)
          
          # Compute the probabilities of real and fake using the Discriminator
          Y_fake = discriminator_model(H_hat)
          Y_fake_e = discriminator_model(E_hat)
          
          # 1. Generator - Adversarial loss
          G_loss_U = loss_object_adversarial(tf.ones_like(Y_fake), Y_fake)
          G_loss_U_e = loss_object_adversarial(tf.ones_like(Y_fake_e), Y_fake_e)
          
          # 2. Generator - Supervised loss
          G_loss_S = loss_object(H[:, 1:, :], H_hat_supervise[:, 1:, :])
          
          # 3. Generator - Two moments (Moment matching)
          G_loss_V1 = tf.reduce_mean(tf.math.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - \
                                            tf.sqrt(tf.nn.moments(X_train,[0])[1] + 1e-6)))
          
          G_loss_V2 = tf.reduce_mean(tf.math.abs((tf.nn.moments(X_hat,[0])[0]) - \
                                            (tf.nn.moments(X_train,[0])[0])))
          
          G_loss_V = G_loss_V1 + G_loss_V2
          
          # Summation of every component of the generator loss
          G_loss = G_loss_U + gamma*G_loss_U_e + 1000*tf.sqrt(G_loss_S) + 100*G_loss_V 
          
        # Compute the gradients with respect to the generator and supervisor model
        gradients_generator=tape.gradient(G_loss,
                                          generator_model.trainable_variables + 
                                          supervisor_model.trainable_variables)
        
        # Apply the gradients to the generator and supervisor model
        optimizer.apply_gradients(zip(gradients_generator, 
                                      generator_model.trainable_variables + 
                                      supervisor_model.trainable_variables))
    
        # Compute the individual components of the generator loss
        g_loss_u(G_loss_U)
        g_loss_v(G_loss_V)
        g_loss_s(G_loss_S)
    
    @tf.function
    def train_step_jointly_embedder(X_train):
        with tf.GradientTape() as tape:
          # Apply Embedder to data and recover the data from the embedding space
          H = embedder_model(X_train) 
          X_tilde = recovery_model(H)
          
          # Compute the loss function for the embedder-recovery model
          r_loss_train = loss_object(X_train, X_tilde)  
          
          # Include the supervision loss but only for 10 %
          H_hat_supervise = supervisor_model(H)
          G_loss_S = loss_object(H[:,1:,:], H_hat_supervise[:,1:,:])
          E_loss = r_loss_train + 0.1 * G_loss_S
          
          # Compute the gradients with respect to the embedder-recovery model
          gradients_embedder=tape.gradient(E_loss,
                                           embedder_model.trainable_variables + 
                                           recovery_model.trainable_variables)
         
          optimizer.apply_gradients(zip(gradients_embedder,
                                        embedder_model.trainable_variables + 
                                        recovery_model.trainable_variables))
    
        # Compute the embedding-recovery loss for training dataset
        e_loss_T0(r_loss_train) 
    
    @tf.function
    def train_step_discriminator(X_train, Z):
        with tf.GradientTape() as tape:
            # Compute embedding for data and random source Z (Wiener process)
            H = embedder_model(X_train) 
            E_hat = generator_model(Z)
            
            # Apply the supervisor model on the random embedding
            H_hat = supervisor_model(E_hat)
            
            # Compute the probabilities of real and fake using the Discriminator
            Y_fake = discriminator_model(H_hat)
            Y_real = discriminator_model(H)     
            Y_fake_e = discriminator_model(E_hat)
                    
            # Loss for the discriminator
            D_loss_real = loss_object_adversarial(tf.ones_like(Y_real), Y_real)
            D_loss_fake = loss_object_adversarial(tf.zeros_like(Y_fake), Y_fake)
            D_loss_fake_e = loss_object_adversarial(tf.zeros_like(Y_fake_e), Y_fake_e)
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
    
        # Compute the gradients with respect to the discriminator model
        gradients_discriminator = tape.gradient(D_loss,
                                                discriminator_model.trainable_variables)
        
        # Apply the gradient to the discriminator model
        optimizer.apply_gradients(zip(gradients_discriminator,
                                      discriminator_model.trainable_variables))
        
        # Compute the discriminator loss
        d_loss(D_loss)    
         
    # Helper counter for the already performed epochs
    already_done_epochs = epoch
        
    # Define the algorithm for training jointly
    print('Start joint training')
    for epoch in range(iterations):
        # Reset the loss at every epoch
        g_loss_u.reset_states()
        g_loss_s.reset_states()
        g_loss_v.reset_states()
        e_loss_T0.reset_states()
        d_loss.reset_states()
        
        # Create two generator and embedding iterations. Just like paper
        for kk in range(2): # Make a random generation
            Z_minibatch = RandomGenerator(batch_size, [20, hidden_dim])
            
            # Train the generator and embedder sequentially
            for x_train in X_train:
                train_step_jointly_generator(x_train, Z_minibatch)
                train_step_jointly_embedder(x_train)
       
        # Train discriminator if too bad or at initialization (0.0)
        if d_loss.result() > 0.15 or d_loss.result() == 0.0:
            # Train the discriminator to optimality
            # In order to be optimizing the Jensen-Shannon divergence
            for i in range(5):
                Z_mb = RandomGenerator(batch_size, [20, hidden_dim])
                for x_train in X_train: # Train the discriminator for 5 epochs
                    train_step_discriminator(x_train, Z_mb)
        
        with summary_writer.as_default():
            tf.summary.scalar('simul/g_loss_u', g_loss_u.result(), step=epoch)
            tf.summary.scalar('simul/g_loss_v', g_loss_v.result(), step=epoch)
            tf.summary.scalar('simul/g_loss_s', g_loss_s.result(), step=epoch)
            tf.summary.scalar('simul/e_loss_T0', e_loss_T0.result(), step=epoch)
            tf.summary.scalar('simul/d_loss', d_loss.result(), step=epoch)
            
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
            
        # Checkpoints
        if epoch % 1 == 0:
            print('step: '+ str(epoch+1) + 
                  ', d_loss: ' + str(np.round(d_loss.result(),4)) + 
                  ', g_loss_u: ' + str(np.round(g_loss_u.result(),4)) + 
                  ', g_loss_s: ' + str(np.round(np.sqrt(g_loss_s.result()),4)) + 
                  ', g_loss_v: ' + str(np.round(g_loss_v.result(),4)) + 
                  ', e_loss_t0: ' + str(np.round(np.sqrt(e_loss_T0.result()),4))  )    
        
    print('Finish joint training')
 
    from data_loading import create_dataset
    # Check if the ESTER rate is equivalent to the EONIA rate
    ester = create_dataset(df='ESTER', normalization='min-max',
                           seq_length=20, training=False)   
    
    # Only get the actual ESTER data and not the additional features
    ester = ester[:, :, 4]
    
    H_hat = embedder_model(ester)
    probs = discriminator_model(H_hat).numpy()
    
    # Check which latent factors 
    
    # Simulate N trajectories of the short rate
    N = 100000
    Z_mb = RandomGenerator(N, [20, 4])
    X_hat_scaled = recovery_model(generator_model(Z_mb)).numpy()
    
    from data_loading import rescale
    
    X_hat = rescale('pre-ESTER', N, 20, X_hat_scaled)
    
    # 4. Train on Synthetic, Test on Real
    from TSTR import value_at_risk
    
    VaR = value_at_risk(X_hat = X_hat, percentile = 99, upper = True)
