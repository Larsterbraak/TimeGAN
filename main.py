"""
MSc Thesis Quantitative Finance
Title: Interest rate risk due to EONIA-ESTER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: June 11th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Inputs

Outputs

"""

import numpy as np
import os
import tensorflow as tf
import pandas as pd

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
parameters = {'hidden_dim':4, 'num_layers':3, 'iterations':310,
              'batch_size': 50, 'module_name':'lstm', 'z_dim':5}

from tgan import run
run(parameters, hparams, X_train, X_test, 
    load=False, load_epochs=0, load_log_dir = '')

# 4. Perform the Train on Synthetic, Test on Real
from metrics import load_models, coverage_test_basel, coverage_test_vasicek, ester_classifier

e_model, r_model, s_model, g_model, d_model = load_models(200) # Load pre-trained models

# Perform the coverage test for a lower and upper Value-at-Risk
classification_lower, exceedances = coverage_test_basel(generator_model = g_model,
                                                        recovery_model = r_model,
                                                        lower=True, 
                                                        hidden_dim = 4)

classification_upper, exceedances = coverage_test_basel(generator_model = g_model,
                                                        recovery_model = r_model,
                                                        lower=False, 
                                                        hidden_dim = 4)

exceedances_upper, exceedances_lower = coverage_test_vasicek()

# Calculate prob of ESTER for TimeGAN calibrated on EONIA 
probs_ester = ester_classifier(load_epochs=50)    

from stylized_facts import descriptives_over_time
sf_over_time = descriptives_over_time()

# Check the correlation matrix 
XY = pd.DataFrame(sf_over_time)
XY.columns = ['V', 'S', 'K', 'H', 'A', 'FT', 'FS', 'SP']
XY['P'] = pd.DataFrame(probs_ester[148:])
import seaborn as sns
corr = XY.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# Skip the first 150 days
import statsmodels.api as sm
sf_over_time = sm.add_constant(sf_over_time)

model = sm.OLS(probs_ester[148:], sf_over_time)
results = model.fit()
print(results.summary())
print(results.summary().as_latex())

# =============================================================================
# Provide a way to adjust the latent space to account for the difference in ESTER and EONIA
# =============================================================================

# Still not done
    
# =============================================================================
# Create a HParams dashboard for hyper parameter tuning
# =============================================================================

from tensorboard.plugins.hparams import api as hp

HP_LR = hp.HParam('learning_rate', hp.Discrete([0.0001, 0.001, 0.01]))
HP_Wasserstein = hp.HParam('Wasserstein_loss', hp.Discrete([True, False]))
HP_Positive_label_smoothing = hp.HParam('Positive_label_smoothing', hp.Discrete([True, False]))
HP_Feature_matching = hp.HParam('Feature_matching', hp.Discrete([True, False]))
HP_T = hp.HParam('T', hp.Discrete([5, 20, 30]))
# Also look for the impact of eta, lambda and kappa on the the performance

METRIC_EXCEEDANCES_UPPER = 'exceedances_upper'
METRIC_EXCEEDANCES_LOWER = 'exceedances_lower'
METRIC_ACCURACY_FAKE = 'accuracy_fake'
METRIC_ACCURACY_REAL = 'accuracy_real'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
      hparams=[HP_LR, HP_Wasserstein, HP_Positive_label_smoothing, HP_Feature_matching, HP_T],
      metrics=[hp.Metric(METRIC_EXCEEDANCES_UPPER, display_name='exceedances_upper'),
               hp.Metric(METRIC_EXCEEDANCES_LOWER, display_name='exceedances_lower'),
               hp.Metric(METRIC_ACCURACY_FAKE, display_name='accuracy_fake'),
               hp.Metric(METRIC_ACCURACY_REAL, display_name='accuracy_real')]
    )

    for lr in HP_LR.domain.values:
        for ws in HP_Wasserstein.domain.values:
            for pls in HP_Positive_label_smoothing.domain.values:
                for fm in HP_Feature_matching.domain.values:
                    for t in HP_T.domain.values:
                        hparams = {
                            HP_LR: lr,
                            HP_Wasserstein: ws,
                            HP_Positive_label_smoothing: pls,
                            HP_Feature_matching: fm,
                            HP_T: t
                            }
                        print({h.name: hparams[h] for h in hparams})
                        hp.hparams(hparams) # Record values used in trial
                        
                        run(parameters, hparams, X_train, X_test, load = False) # run model for 2 epochs

# =============================================================================
# Use the projector mode in Tensorboard for t-SNE and PCA visualizations
# =============================================================================
from tensorboard.plugins import projector

# Some initial code which is the same for all the variants
def register_embedding(embedding_tensor_name, meta_data_fname, log_dir):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = meta_data_fname
    projector.visualize_embeddings(log_dir, config)

def save_labels_tsv(labels, filepath, log_dir):
    with open(os.path.join(log_dir, filepath), 'w') as f:
        for label in labels:
            f.write('{}\n'.format(label))

# Get the real and 'fake' / generated tensors
real = []
counter = 0
length = 0
dimensionality = 0
for _x in X_train:
    counter += _x.shape[0]
    length = _x.shape[1]
    dimensionality = _x.shape[2] 
    real = np.append(real, _x)

from metrics import load_models
from training import RandomGenerator

e_model, r_model, s_model, g_model, d_model = load_models(50) # Load pre-trained models

real = tf.cast(np.reshape(real, newshape=(counter, length, dimensionality)), dtype = tf.float64)    
fake = tf.reshape(r_model(g_model(RandomGenerator(counter, [20, 4]))), (counter, length, dimensionality))

# Concatenate along the first dimension to ge a new tensor
x = tf.concat([real, fake], axis = 0)

# Reshape back to [7054, 20]
x = tf.reshape(x, (7054, 20))
y = np.append(['real' for x in range(counter)], ['fake' for x in range(counter)])

LOG_DIR ='logs'  # Tensorboard log dir
META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
EMBEDDINGS_TENSOR_NAME = 'embeddings'
EMBEDDINGS_FPATH = os.path.join(LOG_DIR, EMBEDDINGS_TENSOR_NAME + 'model.ckpt')
STEP = 0

register_embedding(EMBEDDINGS_TENSOR_NAME, META_DATA_FNAME, LOG_DIR)
save_labels_tsv(y, META_DATA_FNAME, LOG_DIR)

tensor_embeddings = tf.Variable(x, name=EMBEDDINGS_TENSOR_NAME)
saver = tf.compat.v1.train.Saver([tensor_embeddings])  # Must pass list or dict
saver.save(sess=None, global_step=STEP, save_path=EMBEDDINGS_FPATH)
