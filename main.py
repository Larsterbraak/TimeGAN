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
hparams = [] # Used for hyperparameter tuning
parameters = {'hidden_dim':4, 'num_layers':3, 'iterations':100,
              'batch_size': 25, 'module_name':'lstm', 'z_dim':5}

from tgan import run
run(parameters, hparams, X_train, X_test)

# 4. Perform the Train on Synthetic, Test on Real
from metrics import load_models, coverage_test_basel, ester_classifier

e_model, r_model, g_model, d_model = load_models() # Load pre-trained models

# Perform the coverage test for a lower and upper Value-at-Risk
classification_lower = coverage_test_basel(generator_model = g_model,
                                           recovery_model = r_model,
                                           lower=True, 
                                           hidden_dim = 4)

classification_upper = coverage_test_basel(generator_model = g_model,
                                           recovery_model = r_model,
                                           lower=False, 
                                           hidden_dim = 4)

# Calculate prob of ESTER for TimeGAN calibrated on EONIA 
probs_ester = ester_classifier(embedder_model = e_model,
                               discriminator_model = d_model)    

# The probabilities are some where aroud 0.5 so it can not tell whether
# or not it is different from EONIA

# Here we must know the stylized facts per time step and then we can form a correlation matrix 

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