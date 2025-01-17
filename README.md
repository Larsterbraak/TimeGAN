# TimeGAN for short rates

Welcome to my MSc Thesis for completion of the MSc Quantitative Finance at the Erasmus University Rotterdam. In this study, I simulate 1-day, 10-day, and 20-day short rate paths for EONIA using the TimeGAN model. Next to that, I evaluate the ECB's mapping of EONIA to €STER and check the applicability of TimeGAN for interest rate simulation of €STER, i.e. after the EONIA-€STER transition. See the [Thesis](Thesis.pdf) or check out the rest below!

## Table of contents
* [Results](#results)
* [Reproducing paper](#reproducing-paper)
* [Web application](#web-application)
* [Getting started](#getting-started)
* [Technologies](#technologies)
* [Inspiration](#inspiration)

## Results

Below is the visualization of the results presented in Table 4. 

###### T-VaR(99%) estimate for regular TimeGAN (left) and TimeGAN with PLS+FM (right) during the validation dataset.

![Normal TimeGAN T VaR](Figures/Normal_TimeGAN_T_VaR.gif) ![TimeGAN with PLS+FM T VaR](Figures/PLS_FM_TimeGAN_T_VaR.gif)

Below is the visualization of the results presented in Table 4 and Table 5 for TimeGAN with PLS+FM.

###### 1-day, 10-day, and 20-day VaR(99%) estimates for TimeGAN with PLS+FM during validation and test dataset.

![1 day VaR TimeGAN with PLS+FM](Figures/1_day_VaR_PLS_FM.gif) ![10 day VaR TimeGAN with PLS+FM](Figures/10_day_VaR_PLS_FM.gif) ![20 day VaR TimeGAN with PLS+FM](Figures/20_day_VaR_PLS_FM.gif)

## Reproducing paper

* 4 Training TimeGAN
  * For **CPU** version of TimeGAN, see [scripts/tgan.py](scripts/tgan.py)  
  * For **Multi-GPU** version of TimeGAN, see [LISA/tgan.py](LISA/tgan.py)  
* Visualization of training with Tensorboard
  - [TimeGAN](https://tensorboard.dev/experiment/rCW95sn7TNabbXJY4a1gew)
  - [TimeGAN WGAN-GP](https://tensorboard.dev/experiment/vb0fQUArTyqoNIn8RTBgDA)
  - [TimeGAN PLS](https://tensorboard.dev/experiment/591rUg69R1GriM2cGjlP2Q)
  - [TimeGAN FM](https://tensorboard.dev/experiment/1fQKZdtRTPCED1GsEdpUOg)
  - [TimeGAN PLS+FM](https://tensorboard.dev/experiment/kqNuBA7aR96gB07zuM7z5g)
* 5 Data 
  * To produce Figure 5 until 9, see [plotting.py](scripts/plotting.py)  
  * To produce Table 3, see [stylized_facts.py](scripts/stylized_facts.py)
* 7.1 Model selection
  * To produce Figure 11 until 14, see [autoencoder_training.py](scripts/autoencoder_training.py) and [hyper_and_importance.py](hyper_and_importance.py)  
* 7.2 Coverage test
  * To produce Table 4, see [TimeGAN_kupiec.py](scripts/TimeGAN_kupiec.py), [kalman_filter_vasicek.py](scripts/kalman_filter_vasicek.py), and [variance_covariance.py](scripts/variance_covariance.py) 
* 7.3 Diversity of simulations
  * To produce Figure 22 and 23, see [TimeGAN_kupiec.py](scripts/TimeGAN_kupiec.py)
  * To produce Figures 24, 36 until 43 see [main.py](scripts/main.py)
* 7.4 ECB's proposed mapping
  * To produce Figures 26 until 28, see metrics.py, [stylized_facts.py](scripts/stylized_facts.py) and [main.py](scripts/main.py)

## Web application

Would like to see how the model works? This [web application](https://timegan-short-rates.herokuapp.com/) shows the influence of different hyperparameters and allows you to generate your own EONIA or €STER simulations. **Note that it is still under construction**

## Getting started

To train the TimeGAN model on EONIA data, install the folder locally using npm and run tgan.py:

```
$ cd ../TimeGAN-short-rates
$ npm install tgan.py
$ python tgan.py
>> [step: 1, g_loss_u_e: 0.018, g_loss_s: 0.023, g_loss_s_embedder: 0.021, e_loss_t0: 0.312, d_loss: 0.014]
>> [step: 2, g_loss_u_e: 0.019, g_loss_s: 0.022, g_loss_s_embedder: 0.031, e_loss_t0: 0.314, d_loss: 0.029]
...
```

To simulate 20-day short rate paths of EONIA or €STER, run the following code in python

```python
# Generic packages
import pandas as pd
from sklearn import preprocessing
import tensorflow

# TimeGAN specific functions
from metrics import load_models
from training import RandomGenerator

# Import data and apply min-max transformation
df = pd.read_csv("data/Master_EONIA.csv", sep=";")
df = df.iloc[:, 1:] # Remove the Date variable from the dataset
df.EONIA[1:] = np.diff(df.EONIA) # Make first difference for EONIA
df = df.iloc[1:, :] # Remove the first value
scaler = preprocessing.MinMaxScaler().fit(df) # Perform min-max transformation

# Load the models, simulate scaled short rates and unscale
load_epochs = 8250, hparams = [], hidden_dim = 4, T = 20, nr_simulations = N
_, recovery_model, _, generator_model, _ = load_models(load_epochs, hparams, hidden_dim)
Z_mb = RandomGenerator(N, [T, hidden_dim])
samples = recovery_model(generator_model(Z_mb)).numpy()
reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                 samples.shape[2]))
scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                             samples.shape[1], 
                                             samples.shape[2])))    
```

## Technologies

Project is created with:
* Tensorflow version: 2.2
* Python version: 3.6.0
* Tensorboard version: 2.2
* Plotly Dash 1.16

## Inspiration

This MSc Thesis is inspired on TimeGAN by [@jsyoon0823](https://github.com/jsyoon0823/TimeGAN)
