## Introduction

Welcome to the Github page of the MSc Thesis [Interest rate risk simulation using TimeGAN after EONIA-€STER transition using a macro-finance temporal and latent representation based Generative Adversarial Network][Thesis_Lars_ter_Braak_v1.0.] commissioned by the Erasmus University Rotterdam for the completion of the MSc Quantitative Finance. In this study, I try to simulate short rate paths for the EONIA and based on the Discriminator in the TimeGAN evaluate the ECB's mapping of EONIA to €STER.


If you want to run the TimeGAN:

```python
import tgan
tgan()
```

Here you see the GIF for the T-VaR(99%) estimate for regular TimeGAN:

![](Figures/Normal_TimeGAN_T_VaR.gif)

Here you see the GIF for the T-VaR(99%) estimate for TimeGAN with Feature Matching + Positive Label Smoothing:

![](Figures/PLS_FM_TimeGAN_T_VaR.gif)

Here you see the visualizations of 1-day, 10-day, and 20-day VaR(99%) estimates for TimeGAN with Feature Matching + Positive Label smoothing during the validation and test dataset. The upper and lower exceedances and the diversity are based on the combination of the validation and test dataset.

![](Figures/1_day_VaR_PLS_FM.gif)

![](Figures/10_day_VaR_PLS_FM.gif)

![](Figures/20_day_VaR_PLS_FM.gif)


TimeGAN for short rates - Lars ter Braak - MSc Thesis Quantitative Finance

The outline of the report is as follows. 

## 4. Training TimeGAN

For CPU Tensorflow 2 version of TimeGAN, see tgan.py
For Multi-GPU Tensorflow 2 version of TimeGAN, see LISA source code/tgan.py 

## 5. Data 

To produce Figure 5 to 9, see plotting.py
To produce Table 3, see stylized_facts.py

## 7.1 Model selection

To produce Figure 11 to 14, see autoencoder_training.py and hyper_and_importance.py

## 7.2 Coverage test

To produce Table 4, see TimeGAN_kupiec.py, kalman_filter_vasicek.py, and variance_covariance.py
All scripts are a bit entangled. Must get them more out of each other.
