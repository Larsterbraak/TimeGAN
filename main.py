"""
MSc Thesis Quantitative Finance
Title: Interest rate risk due to EONIA-ESTER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: May 27th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Inputs

Outputs

"""

import numpy as np
import os
import datetime
import tensorflow as tf
from tensorboard.plugins import projector

os.chdir('C://Users/s157148/Documents/Github/TimeGAN')

# 1. Create visualization of data
from plotting import plot
plot(hist = True, history = True, pre_ester = True)

# 2. Data loading
from data_loading import create_dataset
X_train, X_test = create_dataset(name = 'EONIA',
                                 normalization = 'min-max',
                                 seq_length = 20,
                                 multidimensional=False)

# 3. Train TimeGAN model
from models.Embedder import Embedder
from models.Recovery import Recovery
from models.Generator import Generator
from models.Supervisor import Supervisor
from models.Discriminator import Discriminator
from training import add_hist, RandomGenerator

hparams = []

parameters = {'hidden_dim':4,
              'num_layers':3,
              'iterations':35,
              'batch_size': 25,
              'module_name':'lstm',
              'z_dim':5}

def run(parameters, hparams):
    
    # Network Parameters
    hidden_dim   = parameters['hidden_dim']  #
    num_layers   = parameters['num_layers']  # Still have to implement
    iterations   = parameters['iterations']  # Test run to check for overfitting
    batch_size   = parameters['batch_size']  # Currently locked at 25
    module_name  = parameters['module_name'] # 'lstm' or 'GRU''
    z_dim        = parameters['z_dim']       # Currently locked at 5
    gamma        = 0
    
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
    g_loss_u_e = tf.keras.metrics.Mean(name='g_loss_u_e')
    g_loss_s = tf.keras.metrics.Mean(name='g_loss_s')
    #g_loss_v = tf.keras.metrics.Mean(name='g_loss_v')
    e_loss_T0 = tf.keras.metrics.Mean(name='e_loss_T0')
    g_loss_s_embedder = tf.keras.metrics.Mean(name='g_loss_s_embedder')
    
    # Create the loss object, optimizer, and training function
    loss_object = tf.keras.losses.MeanSquaredError()
    loss_object_adversarial = tf.losses.BinaryCrossentropy(from_logits=True)
    # from_logits = True because the last dense layers is linear and
    # does not have an activation -- could be differently specified
    optimizer = tf.keras.optimizers.Adam(0.001)
    
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
        
        print(r_loss_train)
        print(r_loss_train.shape)
        
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
            tf.summary.scalar('1. Autoencoder training/', 
                              train_loss.result(),step=epoch)
            tf.summary.scalar('1. Autoencoder training/', 
                              test_loss.result(), step=epoch)
            if epoch % 10 == 0: # Only log trainable variables per 10 epochs
                add_hist(embedder_model.trainable_variables, epoch)
                add_hist(recovery_model.trainable_variables, epoch)
        
        # Log the progress to the user console in python    
        template = 'training: Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch+1, 
                              np.round(train_loss.result().numpy(),8),
                              np.round(test_loss.result().numpy(), 8)))
    
    print('Finished Embedding Network Training')

    # 2. Continue w/ supervisor training on real data (same temporal relations)
    @tf.function
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
      
    @tf.function
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
        
        with summary_writer.as_default():
            tf.summary.scalar('2. Temporal relations training/', 
                              g_loss_s_train.result(), step=epoch)
            tf.summary.scalar('2. Temporal relations training/',
                              g_loss_s_test.result(), step=epoch)
            
            if epoch % 10 == 0: # Only log trainable variables per 10 epochs
                add_hist(generator_model.trainable_variables, epoch)
                add_hist(supervisor_model.trainable_variables, epoch)
                
        template = 'Epoch {}, Train Loss: {}, Test loss: {}'
        print(template.format(epoch+6, 
                              np.round(g_loss_s_train.result().numpy(),8),
                              np.round(g_loss_s_test.result().numpy(),8) ) )
    print('Finished training with Supervised loss only')
    
    # 3. Continue with joint training
    @tf.function # This is 
    def train_step_jointly_generator(X_train, Z):
        with tf.GradientTape() as tape:
          # Apply Embedder to data and apply Supervisor 
          #H = embedder_model(X_train) 
          #H_hat_supervise = supervisor_model(H)
          
          # Apply Generator to Wiener process and create synthetic data
          E_hat = generator_model(Z)
          #X_hat = recovery_model(E_hat)
          
          # Apply Supervisor on fakely generated embeddings
          H_hat = supervisor_model(E_hat)
          
          # Compute the probabilities of real and fake using the Discriminator
          #Y_fake = discriminator_model(H_hat)
          Y_fake_e = discriminator_model(E_hat)
          
          # We do a gradient ascent here because that is easier
          # 1. Generator - Adversarial loss
          #G_loss_U = loss_object_adversarial(tf.ones_like(Y_fake), Y_fake)
          G_loss_U_e = loss_object_adversarial(tf.ones_like(Y_fake_e), Y_fake_e)
          
          # 2. Generator - Supervised loss for fake embeddings
          G_loss_S = loss_object(E_hat[:, 1:, :], H_hat[:, 1:, :])
          
          # 3. Generator - Two moments (Moment matching)
          #G_loss_V1 = tf.reduce_mean(tf.math.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - \
          #                                  tf.sqrt(tf.nn.moments(X_train,[0])[1] + 1e-6)))
          
          #G_loss_V2 = tf.reduce_mean(tf.math.abs((tf.nn.moments(X_hat,[0])[0]) - \
          #                                  (tf.nn.moments(X_train,[0])[0])))
          
          #G_loss_V = G_loss_V1 + G_loss_V2
          
          # Summation of every component of the generator loss
          G_loss = G_loss_U_e + 100*tf.sqrt(G_loss_S) #gamma * G_loss_U_e + 100*tf.sqrt(G_loss_S) + 1000*G_loss_V 
          
        # Compute the gradients with respect to the generator and supervisor model
        gradients_generator=tape.gradient(G_loss,
                                          generator_model.trainable_variables) #+ 
                                          #supervisor_model.trainable_variables)
        
        # Apply the gradients to the generator and supervisor model
        optimizer.apply_gradients(zip(gradients_generator, 
                                      generator_model.trainable_variables)) # + 
                                      #supervisor_model.trainable_variables))
    
        # Compute the individual components of the generator loss
        g_loss_u_e(G_loss_U_e)
        #g_loss_v(G_loss_V)
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
          G_loss_S_embedder = loss_object(H[:,1:,:], H_hat_supervise[:,1:,:])
          
          # Combine the two losses
          E_loss = r_loss_train + 0.1 * G_loss_S_embedder
          
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
    
    @tf.function
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
        #g_loss_v.reset_states()
        e_loss_T0.reset_states()
        g_loss_s_embedder.reset_states()
        
        d_loss.reset_states()
        
        # This for loop is the generator training
        # Create two generator and embedding training iters. Just like paper.
        for kk in range(2): # Make a random generation
            Z_minibatch = RandomGenerator(batch_size, [20, hidden_dim])
                
            # Train the generator and embedder sequentially
            for x_train in X_train:
                train_step_jointly_generator(x_train, Z_minibatch)
                train_step_jointly_embedder(x_train)
       
        # This for loop is the discriminator training
        # Train discriminator if too bad or at initialization (0.0)
        if d_loss.result() > 0.15 or d_loss.result() == 0.0:
            # Train discriminator to optimality (Jensen-Shannon divergence)
            for kk in range(5):
                Z_mb = RandomGenerator(batch_size, [20, hidden_dim])
                for x_train in X_train: # Train the discriminator for 5 epochs
                    train_step_discriminator(x_train, Z_mb)
            
        with summary_writer.as_default():
            # Log the GAN losses
            tf.summary.scalar('3. Joint training/GAN loss/generator',
                              g_loss_u_e.result(), step=epoch)
            tf.summary.scalar('3. Joint training/GAN loss/discriminator', 
                              d_loss.result(), step=epoch)
            
            # Log the temporal relations losses
            tf.summary.scalar('3. Joint training/temporal relations/recurrent relations',
                              g_loss_s.result(), step=epoch)
            
            # Log the autoencoder losses
            tf.summary.scalar('3. Joint training/autoencoder loss/embedding', 
                              e_loss_T0.result(), step=epoch)
            tf.summary.scalar('3. Joint training/autoencoder loss/supervisor',
                              g_loss_s_embedder.result(), step=epoch)
            
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
                  ', g_loss_u_e: ' + str(np.round(g_loss_u_e.result().numpy(),8)) + 
                  ', g_loss_s: ' + str(np.round(np.sqrt(g_loss_s.result().numpy()),8)) + 
                  #', g_loss_v: ' + str(np.round(g_loss_v.result().numpy(),8)) + 
                  
                  ', g_loss_s_embedder: ' + 
                  str(np.round(g_loss_s_embedder.result().numpy(),8)) +
                  ', e_loss_t0: ' + str(np.round(np.sqrt(e_loss_T0.result().numpy()),8)) + 
                  ', d_loss: ' + str(np.round(d_loss.result().numpy(),8)))    
            
    print('Finish joint training')
    
    
    # # Check which latent factors are responsible for the 
    # unravel_latent = np.reshape(H_hat, (H_hat.shape[0] * H_hat.shape[1], H_hat.shape[2]))
    # unravel_probs = np.ravel(probs)    
    
    # from sklearn import linear_model
    # regr = linear_model.LinearRegression()
    # regr.fit(unravel_latent, unravel_probs)
    
    # print('Intercept: \n', regr.intercept_)
    # print('Coefficients: \n', regr.coef_)
    
    # import statsmodels.api as sm
    # X = sm.add_constant(unravel_latent) # adding a constant
    # model = sm.OLS(unravel_probs, X).fit()    
    # model.summary()
    
    from metrics import coverage_test_basel, ester_classifier
    classification_lower = coverage_test_basel(RandomGenerator,
                                               generator_model,
                                               recovery_model,
                                               lower=True, 
                                               hidden_dim = 4)
    
    classification_upper = coverage_test_basel(RandomGenerator,
                                               generator_model,
                                               recovery_model,
                                               lower=False, 
                                               hidden_dim = 4)
    
    probs_ester = ester_classifier(embedder_model, discriminator_model)    
    
# =============================================================================
# Create a HParams dashboard for hyper parameter tuning
# =============================================================================

from tensorboard.plugins.hparams import api as hp

HP_LR = hp.HParam('learning_rate', hp.Discrete([0.0001, 0.001, 0.01]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))

METRIC_D_LOSS = 'd_loss'
METRIC_G_LOSS_U = 'g_loss_u'
METRIC_G_LOSS_V = 'g_loss_v'
METRIC_G_LOSS_S = 'g_loss_s'
METRIC_E_LOSS = 'e_loss'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_LR, HP_DROPOUT],
    metrics=[hp.Metric(METRIC_D_LOSS, display_name='d_loss'),
             hp.Metric(METRIC_G_LOSS_U, display_name='g_loss_u'),
             hp.Metric(METRIC_G_LOSS_V, display_name='g_loss_v'),
             hp.Metric(METRIC_G_LOSS_S, display_name='g_loss_s'),
             hp.Metric(METRIC_E_LOSS, display_name='e_loss')],
  )

for lr in HP_LR.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        hparams = {
            HP_LR: lr,
            HP_DROPOUT: dropout_rate,            
            }
        print({h.name: hparams[h] for h in hparams})
        hp.hparams(hparams) # Record values used in trial
        tgan(hparams = hparams, iterations = 2) # run model for 2 epochs
       
# Remove backslash continuation should solve the problem
# of tensorflow:AutoGraph warning message

# =============================================================================
# Use the projector mode in Tensorboard for t-SNE and PCA visualizations
# =============================================================================

# Some initial code which is the same for all the variants
def register_embedding(embedding_tensor_name, meta_data_fname, log_dir):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_tensor_name
    embedding.metadata_path = meta_data_fname
    projector.visualize_embeddings(log_dir, config)

def save_labels_tsv(labels, filepath, log_dir):
    with open(os.path.join(log_dir, filepath), 'w') as f:
        for label in labels:
            f.write('{}\n'.format(label))

# Get the real and 'fake' / generated tensors
real = tf.reshape(x_train, shape=(500,5))
fake = tf.reshape(recovery_model(generator_model(RandomGenerator(25,
                                                                 [20, 4]))),
                  shape =(500, 5))

# Concatenate along the first dimension to ge a new tensor
x = tf.concat([real, fake], axis = 0)
y = np.append(['real' for x in range(500)], ['fake' for x in range(500)])

LOG_DIR ='C:/Users/s157148/Documents/Research/logs'  # Tensorboard log dir
META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
EMBEDDINGS_TENSOR_NAME = 'embeddings'
EMBEDDINGS_FPATH = os.path.join(LOG_DIR, EMBEDDINGS_TENSOR_NAME + '.ckpt')
STEP = 0

register_embedding(EMBEDDINGS_TENSOR_NAME, META_DATA_FNAME, LOG_DIR)
save_labels_tsv(y, META_DATA_FNAME, LOG_DIR)

# Size of files created on disk: 80.5kB
tensor_embeddings = tf.Variable(x, name=EMBEDDINGS_TENSOR_NAME)
saver = tf.compat.v1.train.Saver([tensor_embeddings])  # Must pass list or dict
saver.save(sess=None, global_step=STEP, save_path=EMBEDDINGS_FPATH)