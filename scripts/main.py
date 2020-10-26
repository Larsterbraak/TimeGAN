"""
MSc Thesis Quantitative Finance
Title: Interest rate risk due to EONIA-ESTER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: October 25th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Inputs

Outputs

"""

import os
os.chdir('C:/Users/s157148/Documents/GitHub/TimeGAN')

import numpy as np
import tensorflow as tf
import pandas as pd

# 1.1 Create visualizations of the data
from plotting import plot
plot(hist=True, history=True, pre_ester=True, spreads=True, inf_gdp=True)

# 1.2 Check the stylized facts of the data
#from stylized_facts import descriptives_over_time
#descriptives_over_time()

# 2. Data loading
from data_loading import create_dataset
X_train, X_test = create_dataset(name = 'EONIA',
                                 normalization = 'min-max',
                                 seq_length = 20,
                                 multidimensional=True)

# 3. Train TimeGAN model
hparams = [] # Used for hyperparameter tuning
parameters = {'hidden_dim':4, 'num_layers':3, 'iterations':5,
              'batch_size': 32, 'module_name':'lstm', 'z_dim':3}

from tgan import run
run(parameters, hparams, X_train, X_test, 
    load=False, load_epochs=0, load_log_dir = '')

# 4. Perform the Train on Synthetic, Test on Real
#from metrics import load_models, coverage_test_basel, coverage_test_vasicek, ester_classifier

#e_model, r_model, s_model, g_model, d_model = load_models(200) # Load pre-trained models

# Perform the coverage test for a lower and upper Value-at-Risk
# classification_lower, exceedances = coverage_test_basel(generator_model = g_model,
#                                                         recovery_model = r_model,
#                                                         lower=True, 
#                                                         hidden_dim = 3)

# classification_upper, exceedances = coverage_test_basel(generator_model = g_model,
#                                                         recovery_model = r_model,
#                                                         lower=False, 
#                                                         hidden_dim = 3)

# exceedances_upper, exceedances_lower = coverage_test_vasicek()

# Calculate prob of ESTER for TimeGAN calibrated on EONIA 
from metrics import ester_classifier
probs_ester = ester_classifier(load_epochs=8250, hparams=[], hidden_dim=4)    

from stylized_facts import descriptives_over_time
sf_over_time = descriptives_over_time()

# Check the correlation matrix 
XY = pd.DataFrame(sf_over_time)
XY.columns = ['V', 'S', 'K', 'H', 'A', 'FT', 'FS', 'SP']
XY['r'] = pd.DataFrame(probs_ester[148:])
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

# # =============================================================================
# # Use the projector mode in Tensorboard for t-SNE and PCA visualizations
# # =============================================================================
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

# Define the settings
hparams = []
hidden_dim = 3
load_epochs = 150

e_model, r_model, s_model, g_model, d_model = load_models(load_epochs, hparams, hidden_dim) # Load pre-trained models

real = tf.cast(np.reshape(real, newshape=(counter, length, dimensionality)), dtype = tf.float64)    
# Embed the real data vectors
real = e_model(real)

# Import all models
fake = tf.reshape(g_model(RandomGenerator(counter, [20, hidden_dim])), (counter, length, hidden_dim))

# Concatenate along the first dimension to ge a new tensor
x = tf.concat([real, fake], axis = 0)

# Reshape back to [2*counter, 20*hidden_dim]
x = tf.reshape(x, (2*counter, 20*hidden_dim))
y = np.append(['real' for x in range(counter)], ['fake' for x in range(counter)])

# LOG_DIR ='logs/WGAN_GP_final'  # Tensorboard log dir
# META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
# EMBEDDINGS_TENSOR_NAME = 'embeddings'
# EMBEDDINGS_FPATH = os.path.join(LOG_DIR, EMBEDDINGS_TENSOR_NAME + 'model.ckpt')
# STEP = load_epochs

# register_embedding(EMBEDDINGS_TENSOR_NAME, META_DATA_FNAME, LOG_DIR)
# save_labels_tsv(y, META_DATA_FNAME, LOG_DIR)

# tensor_embeddings = tf.Variable(x, name=EMBEDDINGS_TENSOR_NAME)
# saver = tf.compat.v1.train.Saver([tensor_embeddings])  # Must pass list or dict
# saver.save(sess=None, global_step=STEP, save_path=EMBEDDINGS_FPATH)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

tsne = TSNE(n_components=2, verbose=1, perplexity=40.0, n_iter=300, n_jobs=-1)
tsne_results = tsne.fit_transform(X=x)

feat_cols = ['point'+str(i) for i in range(x.shape[1])]
df = pd.DataFrame(x.numpy(), columns=feat_cols)
df['y'] = y
df['tsne-one'] = tsne_results[:,0]
df['tsne-two'] = tsne_results[:,1]

plt.figure(dpi=600, figsize=(16,10))
ax=sns.scatterplot(
        x='tsne-one', y='tsne-two',
        hue='y',
        palette=['dodgerblue', 'red'],
        data=df,
        legend="full",
        alpha=1,
        s=30
    )
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])
plt.setp(ax.get_legend().get_texts(), fontsize='26')
plt.axis('off')
plt.show()