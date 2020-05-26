"""
Python script that will do: 
    
    TimeGAN for simulation of €STER short rate based on EONIA rate
    Programmed in Tensorflow 2.1.0.
    
Created on Fri May  1 10:29:49 2020
@author: G.T.F. (Lars) ter Braak
"""

# =============================================================================
# Packages and background
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras import Model
from tensorboard.plugins import projector
from sklearn.preprocessing import MinMaxScaler

# More efficient and elegant pyplot
# plt.style.use(['ieee'])
plt.style.use(['science', 'no-latex'])

# Destroys the current graph and the current session
tf.keras.backend.clear_session()

# Change all layers to have dtype float64
tf.keras.backend.set_floatx('float64')

# =============================================================================
# Importing the used dataset 
# =============================================================================

# Change the working directory and import the data
os.chdir('C:/Users/s157148/Documents/Research')

# To import the PRE-ESTER dataset
df = pd.read_csv('Data/df_full_pre_ester.csv', sep = ';')

# Reverse the dataframe to be chronolocal
df = df.iloc[::-1]

# To import the EONIA dataset
df_EONIA = pd.read_csv("Data/EONIA_rate.csv", sep=";")

# To import the ESTER dataset
df_ester = pd.read_csv("C:/Users/s157148/Documents/Research/Data/ESTER_rate.csv", sep=";")

# Aligning the dates
df_ester = df_ester.iloc[:115, :]

# =============================================================================
# Show the data
# =============================================================================

plt.figure(figsize=(12,8))
plt.hist(np.diff(df_EONIA.EONIA.values[4662:5426]), bins = 75, 
         facecolor = '#0C5DA5', edgecolor = '#169acf',
         linewidth=0.5, density = True, alpha = 0.7,
         label = r'EONIA')

plt.hist(np.append(np.diff(df.WT), np.diff(df_ester.ESTER.values)), bins = 75,
         facecolor = '#00B945', edgecolor = '#169acf',
         linewidth=0.1, density = True, alpha = 0.7, 
         label = r'pre-€STER & €STER')

plt.title(r'Histogram daily difference in $r_t$ during pre-€STER and €STER period')
plt.xlabel(r'Daily difference in $r_t$')
plt.ylabel(r'P(X = x)')
plt.legend(fontsize = 'xx-large', 
           #fancybox = True, shadow=True,
           )#bbox_to_anchor=(0.5, -0.05), ncol = 2)
plt.show()
    
dates = pd.date_range(start = '04-01-1999',
                      end = '12-03-2020',
                      periods = df_EONIA.EONIA.shape[0])
dates_1 = pd.date_range(start = '15-03-2017',
                        end = '30-09-2019',
                        periods = df.shape[0])
dates_2 = pd.date_range(start = '10-01-2019',
                        end = '12-03-2020',
                        periods = df_ester.shape[0])

plt.figure(figsize=(12,8))
plt.plot(, df_EONIA.EONIA.values)
plt.plot(dates_1, df.WT.values)
plt.plot(dates_2, df_ester.ESTER.values)
plt.legend(('EONIA', 'pre-€STER', '€STER'), fontsize = 'xx-large')
plt.ylabel(r'Short rate $r_t$ [%]')
plt.xlabel(r'Time $t$')
plt.title(r'Short rates $r_t$ over time')
plt.show()

# Show the EONIA and ESTER at the same time
plt.figure(figsize=(12,8))
plt.plot()
df_EONIA.Date.values[4662:5426]


dates = pd.date_range(start = '15-03-2017',
                      end = '30-09-2019',
                      periods = df.shape[0])

# The top plot consisting of daily closing prices
top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
top.plot(dates, df.R25, label = r'$25^{th}$ percentile €STER')
top.plot(dates, df.WT, label = r'Weighted trimmed mean €STER')
top.plot(dates, df.R75, label = r'$75^{th}$ percentile €STER')

plt.title('€STER during pre-€STER period')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Short rate $r_t$ [$\%$]')
top.legend(loc = 'best', fontsize = 'xx-large')
# The bottom plot consisting of daily trading volume
bottom = plt.subplot2grid((4, 4), (3,0), rowspan=1, colspan=4)
bottom.plot(dates, df.TT *1e6, color = '#FF2C00', label = r'Transaction volume')
bottom.plot(dates, df.NT *1e8, color = '#845B97', label = r'Nr. of transactions')

def to_transactions(x):
    return x * 1e-8

def to_volume(x):
    return x * 1e8

ax = plt.gca()
secaxy = ax.secondary_yaxis('right', functions = (to_transactions, to_volume))
secaxy.set_ylabel(r'Nr. of transactions')

plt.title('Transaction volume during pre-€STER period')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Volume $[€]$')
plt.ylim((2e10, 9e10))
plt.legend(loc = 'upper right')

plt.gcf().set_size_inches(12, 8)
plt.subplots_adjust(hspace=0.75)

# =============================================================================
# Transform the source data to be used in Tensorflow 2.x
# =============================================================================

def create_dataset(df, seq_length):
    dataX = []
    
    # Make lookback periods of seq_length in the data
    for i in range(0, len(df) - seq_length):
        _df = df[i : i + seq_length]
        dataX.append(_df)
    
    # Create random permutations to make it more i.i.d.
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
        
    return outputX

# Convert the interest rates to daily differences
# We are modelling the interest rate discretized differential just
# as Vasicek, Ho-lee etc
df.iloc[1:,[1,2,4]] = np.diff(df[['R25', 'R75', 'WT']], axis =0)
df = df.iloc[1:]

# Transform the data using a MinMaxScaler
df = MinMaxScaler().fit_transform(df)

# Create the dataset and reshape in the correct format
df = create_dataset(df, 20)
df = np.reshape(df, newshape=(628,20,5))

# Split the data in train and test
X_train = df[0:400,:,:]
X_test = df[400:,:,:]

# Clear up memory space
del df

# Make the data into tensorflow data
X_train = tf.data.Dataset.from_tensor_slices((X_train))
X_train = X_train.batch(25)
X_test = tf.data.Dataset.from_tensor_slices((X_test))
X_test = X_test.batch(25)

# =============================================================================
# Make the customized models (TimeGAN)
# =============================================================================

# Define the hyperparameters to use
gamma = 1

def run(hparams, iterations):
    # Embedder network in Tensorflow 2.x
    class Embedder(Model):
      def __init__(self, tensorboard_folder_path, hparams):
        super(Embedder, self).__init__()
        self.LSTM1 = LSTM(units=10, 
                          return_sequences=True,
                          input_shape=(20,5),
                          kernel_initializer = 'he_uniform',
                          dropout = hparams[HP_DROPOUT],
                          recurrent_dropout = 0.2,
                          name = 'LSTM1')
        self.LSTM2 = LSTM(units=7,
                          return_sequences=True,
                          kernel_initializer = 'he_uniform',
                          dropout = hparams[HP_DROPOUT],
                          recurrent_dropout = 0.2,
                          name = 'LSTM2')
        self.LSTM3 = LSTM(units=4,
                          return_sequences=True,
                          kernel_initializer = 'he_uniform',
                          dropout = hparams[HP_DROPOUT],
                          recurrent_dropout = 0.2,
                          name = 'LSTM3')
        self.Dense1 = Dense(units=4,
                            activation='sigmoid', # To ensure [0, 1]
                            name = 'Dense1')
        self.graph_has_been_written=False
        self.tensorboard_folder_path = tensorboard_folder_path
    
      def call(self, x, **kwargs):
        x = self.LSTM1(x)
        x = self.LSTM2(x)
        x = self.LSTM3(x)
        x = self.Dense1(x)
        
        # Print the graph in TensorBoard
        if not self.graph_has_been_written:
            model_graph = x.graph
            writer = tf.compat.v1.summary.FileWriter(logdir=self.tensorboard_folder_path,
                                                     graph=model_graph)
            writer.flush()
            self.graph_has_been_written = True
            print("Wrote eager graph to:", self.tensorboard_folder_path)
        
        return x
        
    # Create an instance of the embedder model
    embedder_model = Embedder('logs/embedder', hparams)
    
    # Recovery network in Tensorflow 2.x
    class Recovery(Model):
      def __init__(self, tensorboard_folder_path, hparams):
        super(Recovery, self).__init__()
        self.LSTM1 = LSTM(units=10, 
                          return_sequences=True,
                          kernel_initializer = 'he_uniform',
                          dropout = hparams[HP_DROPOUT],
                          recurrent_dropout = 0.2,
                          input_shape=(20,4), 
                          name = 'LSTM1')
        self.LSTM2 = LSTM(units=7, 
                          return_sequences=True,
                          kernel_initializer = 'he_uniform',
                          dropout = hparams[HP_DROPOUT],
                          recurrent_dropout= 0.2,
                          name = 'LSTM2')
        self.LSTM3 = LSTM(units=4,
                          return_sequences=True,
                          kernel_initializer = 'he_uniform',
                          dropout = hparams[HP_DROPOUT],
                          recurrent_dropout = 0.2,
                          name = 'LSTM3')
        self.Dense1 = Dense(units=5, 
                            activation='sigmoid', 
                            name = 'Dense1')
        self.graph_has_been_written=False
        self.tensorboard_folder_path = tensorboard_folder_path
        
      def call(self, x, **kwargs):
        x = self.LSTM1(x)
        x = self.LSTM2(x)
        x = self.LSTM3(x)
        x = self.Dense1(x)  
        
        # Print the graph in TensorBoard
        if not self.graph_has_been_written:
            model_graph = x.graph
            writer = tf.compat.v1.summary.FileWriter(logdir=self.tensorboard_folder_path,
                                                     graph=model_graph)
            writer.flush()
            self.graph_has_been_written = True
            print("Wrote eager graph to:", self.tensorboard_folder_path)
        
        return x
    
    # Create an instance of the recovery model
    recovery_model = Recovery('logs/recovery', hparams)
    
    # Generator network in Tensorflow 2.x
    class Generator(Model):
        def __init__(self, tensorboard_folder_path, hparams):
            super(Generator, self).__init__()
            self.LSTM1 = LSTM(units=10, 
                          return_sequences=True,
                          input_shape=(20,5),
                          kernel_initializer = 'he_uniform',
                          dropout = hparams[HP_DROPOUT],
                          recurrent_dropout = 0.2,
                          name = 'LSTM1')
            self.LSTM2 = LSTM(units=7,
                              return_sequences=True,
                              kernel_initializer = 'he_uniform',
                              dropout = hparams[HP_DROPOUT],
                              recurrent_dropout = 0.2,
                              name = 'LSTM2')
            self.LSTM3 = LSTM(units=4,
                              return_sequences=True,
                              kernel_initializer = 'he_uniform',
                              dropout = hparams[HP_DROPOUT],
                              recurrent_dropout = 0.2,
                              name = 'LSTM3')
            self.Dense1 = Dense(units=4,
                                activation='sigmoid', # To ensure [0, 1]
                                name = 'Dense1')
            self.graph_has_been_written=False
            self.tensorboard_folder_path = tensorboard_folder_path
    
        def call(self, x, **kwargs):
            x = self.LSTM1(x)
            x = self.LSTM2(x)
            x = self.LSTM3(x)
            x = self.Dense1(x)
            
             # Print the graph in TensorBoard
            if not self.graph_has_been_written:
                model_graph = x.graph
                writer = tf.compat.v1.summary.FileWriter(logdir=self.tensorboard_folder_path,
                                                         graph=model_graph)
                writer.flush()
                self.graph_has_been_written = True
                print("Wrote eager graph to:", self.tensorboard_folder_path)
           
            return x
        
    # Create an instance of the generator model
    generator_model = Generator('logs/generator', hparams)
    
    # Supervisor network in Tensorflow 2.x
    class Supervisor(Model):
        def __init__(self, tensorboard_folder_path, hparams):
            super(Supervisor, self).__init__()
            self.LSTM1 = LSTM(units = 7,
                              return_sequences=True,
                              kernel_initializer = 'he_uniform',
                              dropout = hparams[HP_DROPOUT],
                              recurrent_dropout = 0.2,
                              input_shape=(20,4),
                              name = 'LSTM1')
            self.LSTM2 = LSTM(units=6,
                              return_sequences=True,
                              kernel_initializer = 'he_uniform',
                              dropout = hparams[HP_DROPOUT],
                              recurrent_dropout = 0.2,
                              name='LSTM2')
            self.Dense1 = Dense(units=4,
                                activation='sigmoid', # To stay in the [0,1] range
                                name = 'Dense1')
            self.graph_has_been_written=False
            self.tensorboard_folder_path=tensorboard_folder_path
            
        def call(self, x, **kwargs):
            x = self.LSTM1(x)
            x = self.LSTM2(x)
            x = self.Dense1(x)
            
            # Print the graph in TensorBoard
            if not self.graph_has_been_written:
                model_graph = x.graph
                writer = tf.compat.v1.summary.FileWriter(logdir=self.tensorboard_folder_path,
                                                         graph=model_graph)
                writer.flush()
                self.graph_has_been_written = True
                print("Wrote eager graph to:", self.tensorboard_folder_path)
           
            return x
    
    # Create an instance of the supervisor model
    supervisor_model = Supervisor('logs/supervisor', hparams)
    
    # Discriminator network for GAN in latent space in Tensorflow 2.x
    class Discriminator(Model):
        def __init__(self, tensorboard_folder_path, hparams):
            super(Discriminator, self).__init__()
            self.LSTM1 = Bidirectional(LSTM(units=7,
                                            return_sequences=True,
                                            kernel_initializer = 'he_uniform',
                                            dropout = hparams[HP_DROPOUT],
                                            recurrent_dropout = 0.2,
                                            input_shape=(20,4),
                                            name='LSTM1'))
            self.LSTM2 = Bidirectional(LSTM(units=4,
                                            return_sequences=True,
                                            kernel_initializer = 'he_uniform',
                                            dropout = hparams[HP_DROPOUT],
                                            recurrent_dropout = 0.2,
                                            name='LSTM2'))
            self.LSTM3 = Bidirectional(LSTM(units=2,
                                            return_sequences=True,
                                            kernel_initializer = 'he_uniform',
                                            dropout = hparams[HP_DROPOUT],
                                            recurrent_dropout = 0.2,
                                            name='LSTM3'))
            self.Dense1 = Dense(units=1,
                                activation=None,
                                name='Dense1')
            self.graph_has_been_written=False
            self.tensorboard_folder_path = tensorboard_folder_path
            
        def call(self, x):
            x = self.LSTM1(x)
            x = self.LSTM2(x)
            x = self.LSTM3(x)
            x = self.Dense1(x)
            
            # Print the graph in TensorBoard
            if not self.graph_has_been_written:
                model_graph = x.graph
                writer = tf.compat.v1.summary.FileWriter(logdir=self.tensorboard_folder_path,
                                                         graph=model_graph)
                writer.flush()
                self.graph_has_been_written = True
                print("Wrote eager graph to:", self.tensorboard_folder_path)
            
            return x
        
    # Create an instance of the discriminator model
    discriminator_model = Discriminator('logs/discriminator', hparams)
    
    # Random vector generation Z
    def RandomGenerator(batch_size, z_dim):
        Z_minibatch = list()
        
        for i in range(batch_size): 
            Z_minibatch.append(np.random.uniform(0., 1, [z_dim[0], z_dim[1]]))
            
        # Reshape the random matrices
        Z_minibatch = np.reshape(Z_minibatch, (batch_size, z_dim[0], z_dim[1]))
        return Z_minibatch
    
    # =============================================================================
    # Define the TensorBoard such that we can visualize the results
    # =============================================================================
    import datetime
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/ ' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    def add_hist(train_vars, epoch):
        for i in train_vars:
            name = i.name.split(":")[0]
            value = i.value()
            tf.summary.histogram(name, value, step=epoch)
    
    # =============================================================================
    # Create the loss object, optimizer, and training function
    # =============================================================================

    # Create a loss object
    loss_object = tf.keras.losses.MeanSquaredError()
    
    # Create a Adam optimizer
    optimizer = tf.keras.optimizers.Adam(hparams[HP_LR]) # Standard lr
    
    # Metrics to track during training
    train_loss = tf.keras.metrics.Mean(name='train_loss', dtype = tf.float32)
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype = tf.float32)
    
    @tf.function
    def train_step_auto_encode(X_train):
        with tf.GradientTape() as tape:
            # Apply the Embedder to the data
            e_pred_train = embedder_model(X_train)
            
            # Assert to check the shape of the embedding space
            tf.debugging.assert_equal(e_pred_train.shape, (25, 20, 4))
            
            # Apply the Recovery to the predicted hidden states    
            r_pred_train = recovery_model(e_pred_train)
            
            # Assert to check the shape of the recovery space
            tf.debugging.assert_equal(r_pred_train.shape, (25, 20, 5))
            
            # Compute the loss function for the LSTM autoencoder
            r_loss_train = loss_object(X_train, r_pred_train)
            
            # Assert to check if the loss is non-negative
            tf.debugging.assert_non_negative(r_loss_train)
            
        # All forward operations are recorder on a tape
        # In order to provide backpropagation lateer
        
        # Compute the gradients with respect to the Embedder and Recovery vars
        gradients = tape.gradient(r_loss_train, 
                                  embedder_model.trainable_variables \
                                  + recovery_model.trainable_variables)
        
        # Apply the gradients to the Embedder and Recovery vars
        optimizer.apply_gradients(zip(gradients, 
                                      embedder_model.trainable_variables +
                                      recovery_model.trainable_variables))
        train_loss(r_loss_train)
      
    @tf.function
    def test_step_auto_encode(X_test):
        
        # Apply the Embedder to the data
        e_pred_test = embedder_model(X_test)
        
        # Apply the Recovery to the predicted hidden states    
        r_pred_test = recovery_model(e_pred_test)
        
        # Compute the loss function for the LSTM autoencoder
        r_loss_test = loss_object(X_test, r_pred_test)
      
        test_loss(r_loss_test)    
        
    # =============================================================================
    # Start with embedder training (Optimal LSTM auto encoder network)    
    # =============================================================================
    EPOCHS = iterations
    
    # Check whether the training and test set are tensorflow datasets
    assert isinstance(X_train, tf.data.Dataset)
    assert isinstance(X_test, tf.data.Dataset)
    
    # Train the embedder for the input data
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()
        
        # Train over the complete train dataset
        for x_train in X_train:
            train_step_auto_encode(x_train)
        
        # Test over the complete test dataset
        for x_test in X_test:
            test_step_auto_encode(x_test)
        
        with summary_writer.as_default():
            tf.summary.scalar('recovery/train', 
                              train_loss.result(),
                              step=epoch)
            tf.summary.scalar('recovery/train', 
                              test_loss.result(), 
                              step=epoch)
            add_hist(embedder_model.trainable_variables, epoch)
            add_hist(recovery_model.trainable_variables, epoch)
        
        # Log the progress to the user console in python    
        template = 'training: Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch+1, train_loss.result(), test_loss.result() ) )
    print('Finished Embedding Network Training')
    
    # =============================================================================
    # Continu with supervised loss
    # =============================================================================
    
    # Metrics to track during training
    g_loss_s_train = tf.keras.metrics.Mean(name='g_loss_s_train')
    g_loss_s_test = tf.keras.metrics.Mean(name='g_loss_s_test')
    
    @tf.function
    def train_step_supervised(X_train):
      with tf.GradientTape() as tape:
        
        # Apply the Embedder to real train data
        e_pred_train = embedder_model(X_train)
        
        # Check the temporal relations in the supervisor network
        # by trying to predict the embedder structure
        H_hat_supervise = supervisor_model(e_pred_train)
        
        # Compute the squared loss for the real embedding
        # and the supervised embedding
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
        
        # Apply the Embedder to real test data
        e_pred_test = embedder_model(X_test)
        
        # Check temporal relations in embedding space (latent)
        H_hat_supervise_test = supervisor_model(e_pred_test)
        
        # Compute the squared loss for the real test embedding
        # and the supervised test embedding
        G_loss_S = loss_object(e_pred_test[:, 1:, :], 
                               H_hat_supervise_test[:, 1:, :])
            
        g_loss_s_test(G_loss_S)
    
    # =============================================================================
    # Train supervised loss first such that temporal relations are preserved
    # =============================================================================
    
    for epoch in range(EPOCHS):
        g_loss_s_train.reset_states()
        g_loss_s_test.reset_states()
        
        for x_train in X_train:
            train_step_supervised(x_train)
            
        for x_test in X_test:
            test_step_supervised(x_test)
        
        with summary_writer.as_default():
            tf.summary.scalar('generator/train', 
                              g_loss_s_train.result(),
                              step=epoch)
            tf.summary.scalar('generator/test',
                              g_loss_s_test.result(),
                              step=epoch)
            add_hist(generator_model.trainable_variables, epoch)
            add_hist(supervisor_model.trainable_variables, epoch)
                
        template = 'Epoch {}, Loss: {}, Test loss: {}'
        print(template.format(epoch+1, 
                              g_loss_s_train.result(),
                              g_loss_s_test.result() ) )
    
    print('Finished training with Supervised loss only')
    
    # =============================================================================
    # Continue with joint training
    # =============================================================================
    # Create a loss object for the sigmoid cross entropy
    loss_object_adversarial = tf.losses.BinaryCrossentropy(from_logits=True)
    # from_logits = True because the last dense layers is linear and
    # does not have an activation -- could be differently specified
    
    # Metrics to track during training
    d_loss = tf.keras.metrics.Mean(name='d_loss')
    g_loss_u = tf.keras.metrics.Mean(name='g_loss_u')
    g_loss_s = tf.keras.metrics.Mean(name='g_loss_s')
    g_loss_v = tf.keras.metrics.Mean(name='g_loss_v')
    e_loss_T0 = tf.keras.metrics.Mean(name='e_loss_T0')
    
    #test_loss_jointly = tf.keras.metrics.Mean(name='test_loss_jointly_autoencoding')
    
    @tf.function
    def train_step_jointly_generator(X_train, Z):
        with tf.GradientTape() as tape:
          
          # Apply the Embedder to the data
          H = embedder_model(X_train) 
          
          # Generator
          E_hat = generator_model(Z)
          H_hat = supervisor_model(E_hat)
          H_hat_supervise = supervisor_model(H)
          
          # Synthetic data
          X_hat = recovery_model(H_hat)
          
          # Compute the probabilities of real and fake using the Discriminator
          Y_fake = discriminator_model(H_hat)
          Y_fake_e = discriminator_model(E_hat)
          
          # Generator loss
          # 1. Adversarial loss
          G_loss_U = loss_object_adversarial(tf.ones_like(Y_fake), Y_fake)
          G_loss_U_e = loss_object_adversarial(tf.ones_like(Y_fake_e), Y_fake_e)
          
          # Supervised loss
          G_loss_S = loss_object(H[:, 1:, :], H_hat_supervise[:, 1:, :])
          
          # Two moments (Moment matching)
          G_loss_V1 = tf.reduce_mean(tf.math.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - \
                                            tf.sqrt(tf.nn.moments(X_train,[0])[1] + 1e-6)))
          
          G_loss_V2 = tf.reduce_mean(tf.math.abs((tf.nn.moments(X_hat,[0])[0]) - \
                                            (tf.nn.moments(X_train,[0])[0])))
          
          G_loss_V = G_loss_V1 + G_loss_V2
          
          # # Summation
          G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V 
          
        # Compute the gradients with respect to the generator and supervisor model
        gradients_generator = tape.gradient(G_loss,
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
    def train_step_jointly_embedder(X_train, Z):
        with tf.GradientTape() as tape:
          
          # Apply the Embedder to the data
          H = embedder_model(X_train) 
          X_tilde = recovery_model(H)
          
          # Compute the loss function for the embedder model
          r_loss_train = loss_object(X_train, X_tilde)  
          
          # Include the supervision loss
          H_hat_supervise = supervisor_model(H)
          
          G_loss_S = loss_object(H[:,1:,:], H_hat_supervise[:,1:,:])
          
          E_loss = r_loss_train + 0.1 * G_loss_S
          
          # Compute the gradients with respect to the embedder and recovery model
          gradients_embedder = tape.gradient(E_loss,
                                              embedder_model.trainable_variables + 
                                              recovery_model.trainable_variables)
         
          # Apply the gradient to the embedder and recovery model
          optimizer.apply_gradients(zip(gradients_embedder,
                                        embedder_model.trainable_variables + 
                                        recovery_model.trainable_variables))
    
        # Compute the auto encoding loss for the training set
        e_loss_T0(r_loss_train) 
    
    @tf.function
    def train_step_discriminator(X_train, Z):
        with tf.GradientTape() as tape:
            
            # Compute the embedding for the real data
            H = embedder_model(X_train) 
            
            # Generate an embedding for the random source Z
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
    for epoch in range(EPOCHS):
        # Reset the loss at every epoch
        g_loss_u.reset_states()
        g_loss_s.reset_states()
        g_loss_v.reset_states()
        e_loss_T0.reset_states()
        d_loss.reset_states()
        
        # Create two generator and embedding iterations
        for kk in range(2):
            
            # Make a random generation
            Z_minibatch = RandomGenerator(25, [20, 4])
            
            # Train the generator and embedder sequentially
            for x_train in X_train:
                train_step_jointly_generator(x_train, Z_minibatch)
                train_step_jointly_embedder(x_train, Z_minibatch)
       
        # Train discriminator if too bad or at initialization (0.0)
        if d_loss.result() > 0.15 or d_loss.result() == 0.0:
            # Train the discriminator to optimize the Jensen-Shannon divergence
            for i in range(5):
                Z_mb = RandomGenerator(25, [20, 4])
                # Train the discriminator
                for x_train in X_train:
                    train_step_discriminator(x_train, Z_mb)
        
        with summary_writer.as_default():
            tf.summary.scalar('simul/g_loss_u', g_loss_u.result(), step=epoch)
            tf.summary.scalar('simul/g_loss_v', g_loss_v.result(), step=epoch)
            tf.summary.scalar('simul/g_loss_s', g_loss_s.result(), step=epoch)
            tf.summary.scalar('simul/e_loss_T0', e_loss_T0.result(), step=epoch)
            tf.summary.scalar('simul/d_loss', d_loss.result(), step=epoch)
            
        with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
            tf.summary.scalar(METRIC_G_LOSS_U, g_loss_u.result(), step=epoch)
            tf.summary.scalar(METRIC_G_LOSS_V, g_loss_v.result(), step=epoch)
            tf.summary.scalar(METRIC_G_LOSS_S, g_loss_s.result(), step=epoch)
            tf.summary.scalar(METRIC_E_LOSS, e_loss_T0.result(), step=epoch)
            tf.summary.scalar(METRIC_D_LOSS, d_loss.result(), step=epoch)
        
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
            
        #Checkpoints
        if epoch % 1 == 0:
            print('step: '+ str(epoch+1) + 
                  ', d_loss: ' + str(np.round(d_loss.result(),4)) + 
                  ', g_loss_u: ' + str(np.round(g_loss_u.result(),4)) + 
                  ', g_loss_s: ' + str(np.round(np.sqrt(g_loss_s.result()),4)) + 
                  ', g_loss_v: ' + str(np.round(g_loss_v.result(),4)) + 
                  ', e_loss_t0: ' + str(np.round(np.sqrt(e_loss_T0.result()),4))  )    
        
    print('Finish joint training')
  
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
        run(hparams = hparams, iterations = 2) # run model for 2 epochs
       
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