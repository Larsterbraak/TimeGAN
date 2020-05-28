import numpy as np
import os
import datetime
import tensorflow as tf
#from tensorboard.plugins import projector

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
              'iterations':2,
              'batch_size': 25,
              'module_name':'lstm',
              'z_dim':5}

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
summary_writer_train = tf.summary.create_file_writer(log_dir + '/train')
summary_writer_test = tf.summary.create_file_writer(log_dir + '/test')

# Create an instance of all neural networks models (All LSTM)
embedder_model = Embedder('logs/embedder', hparams)
recovery_model = Recovery('logs/recovery', hparams, dimensionality = 1) # If used for EONIA rate only
generator_model = Generator('logs/generator', hparams)
supervisor_model = Supervisor('logs/supervisor', hparams)
discriminator_model = Discriminator('logs/TimeGAN', hparams)

# Metrics to track during training
r_loss_train = tf.keras.metrics.Mean(name='r_loss_train')
r_loss_test = tf.keras.metrics.Mean(name='r_loss_test')
g_loss_s_train = tf.keras.metrics.Mean(name='g_loss_s_train')
g_loss_s_test = tf.keras.metrics.Mean(name='g_loss_s_test')
d_loss = tf.keras.metrics.Mean(name='d_loss')
g_loss_u_e = tf.keras.metrics.Mean(name='g_loss_u_e')
g_loss_s = tf.keras.metrics.Mean(name='g_loss_s')
e_loss_T0 = tf.keras.metrics.Mean(name='e_loss_T0')
g_loss_s_embedder = tf.keras.metrics.Mean(name='g_loss_s_embedder')

# Create the loss object, optimizer, and training function
loss_object = tf.keras.losses.MeanSquaredError()
loss_object_adversarial = tf.losses.BinaryCrossentropy(from_logits=True)
# from_logits = True because the last dense layers is linear and
# does not have an activation -- could be differently specified
optimizer = tf.keras.optimizers.Adam(0.01)

tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()

# 1. Start with embedder training (Optimal LSTM auto encoder network)
@tf.function
def train_step_embedder(X_train):
  with tf.GradientTape() as tape:
      
      # Apply Embedder to data and Recovery to predicted hidden states 
      e_pred_train = embedder_model(X_train)
      r_pred_train = recovery_model(e_pred_train)
  
      # Compute loss for LSTM autoencoder
      R_loss_train = loss_object(X_train, r_pred_train)
  
      # Compute the gradients with respect to the Embedder and Recovery vars
      gradients = tape.gradient(R_loss_train, 
                                embedder_model.trainable_variables +
                                recovery_model.trainable_variables)
  
      # Apply the gradients to the Embedder and Recovery vars
      optimizer.apply_gradients(zip(gradients, 
                                    embedder_model.trainable_variables +
                                    recovery_model.trainable_variables))
  
      r_loss_train(R_loss_train)

@tf.function
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