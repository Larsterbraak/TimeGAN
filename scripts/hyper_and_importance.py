# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:24:28 2020

@author: s157148
"""
import numpy as np
import matplotlib.pyplot as plt

results_dropout_0_1 = [[6.95, 2.19, 1.26, 1.087, 0.923],
                       [9.97, 9.64, 5.914, 5.568, 1.12],
                       [9.27, 9.71, 9.78, 7.78, 9.67]]

results_dropout_0_2 = [[9.97, 9.64, 5.91, 5.57, 1.12],
                       [14.35, 9.87, 10, 10, 1.15],
                       [10, 10, 10, 10, 10]]

results_dropout_0_3 = [[7, 7, 7, 7, 7],
                       [7, 7, 7, 7, 7],
                       [7, 7, 7, 7, 7]]

layers = [1, 2, 3]
latent = [1, 2, 3, 4, 5]

# Specify the droput level and reshape to the grid structure
means = np.reshape(results_dropout_0_1, newshape=(len(layers), len(latent)))

# Make a contour plot of the hyperparametrized performance
xx, yy = np.meshgrid(latent, layers) 
plt.figure(dpi=800)
cs = plt.contourf(xx, yy, means, levels=[3,4,5,6,7,8,9,10])
plt.contourf(xx, yy, means)
plt.ylabel('Number of layers in Autoencoder')
plt.xlabel('Number of latent dimensions')
plt.title('Hyperparameter tuning for Autoencoder network')
proxy = [plt.Rectangle((0,0),1,1, fc=pc.get_facecolor()[0]) for pc in cs.collections]
interval = (np.max(means) - np.min(means)) / 8
plt.legend(proxy, [str(np.round(np.min(means),2)) + "-" + str(np.round(np.min(means) + interval,2)),
                   str(np.round(np.min(means) + interval,2)) + "-" + str(np.round(np.min(means) + 2*interval,2)),
                   str(np.round(np.min(means) + 2*interval,2)) + "-" + str(np.round(np.min(means) + 3*interval,2)),
                   str(np.round(np.min(means) + 3*interval,2)) + "-" + str(np.round(np.min(means) + 4*interval,2)),
                   str(np.round(np.min(means) + 4*interval,2)) + "-" + str(np.round(np.min(means) + 5*interval,2)),
                   str(np.round(np.min(means) + 5*interval,2)) + "-" + str(np.round(np.min(means) + 6*interval,2)),
                   str(np.round(np.min(means) + 6*interval,2)) + "-" + str(np.round(np.min(means) + 7*interval,2))])
plt.show()

# Inspect the performance of different layers
result_layers = [[1.26, 2.80, 2.13],
                 [4.22, 5.914, 7.73],
                 [4.08, 5.66, 9.78]]

layer_1 = [1, 2, 3]
layer_2 = [1, 2, 3]

# Specify the droput level and reshape to the grid structure
means = np.reshape(result_layers, newshape=(len(layer_1), len(layer_2)))

# Make a contour plot of the hyperparametrized performance
xx, yy = np.meshgrid(layer_1, layer_2) 
plt.figure(dpi=800)
cs = plt.contourf(xx, yy, means, levels=[3,4,5,6,7,8,9])
plt.contourf(xx, yy, means)
plt.ylabel('Number of layers in Embedder')
plt.xlabel('Number of layers in Recovery')
plt.title('Hyperparameter tuning for Autoencoder network')
proxy = [plt.Rectangle((0,0),1,1, fc=pc.get_facecolor()[0]) for pc in cs.collections]
interval = (np.max(means) - np.min(means)) / 6
plt.legend(proxy, [str(np.round(np.min(means),2)) + "-" + str(np.round(np.min(means) + interval,2)),
                   str(np.round(np.min(means) + interval,2)) + "-" + str(np.round(np.min(means) + 2*interval,2)),
                   str(np.round(np.min(means) + 2*interval,2)) + "-" + str(np.round(np.min(means) + 3*interval,2)),
                   str(np.round(np.min(means) + 3*interval,2)) + "-" + str(np.round(np.min(means) + 4*interval,2)),
                   str(np.round(np.min(means) + 4*interval,2)) + "-" + str(np.round(np.min(means) + 5*interval,2)),
                   str(np.round(np.min(means) + 5*interval,2)) + "-" + str(np.round(np.min(means) + 6*interval,2))])
plt.show()

from decimal import Decimal

# Hyperparameter optimization for Supervisor network
results_supervisor = [[1.7239057e-4, 1.05683466e-4, 0.00032588467],
                      [0.00021585949, 0.000285094250, 0.00039172117],
                      [0.00038000430, 0.000211731310, 0.00065456104]]

layers = [1, 2, 3]
dropout = [0.1, 0.2, 0.3]

# Specify the droput level and reshape to the grid structure
means = np.reshape(results_supervisor, newshape=(len(layers), len(dropout)))

# Make a contour plot of the hyperparametrized performance
xx, yy = np.meshgrid(dropout, layers) 
plt.figure(dpi=500)
cs = plt.contourf(xx, yy, means, levels=[3,4,5,6,7,8,9])
plt.contourf(xx, yy, means)
plt.ylabel('Number of layers in Supervisor')
plt.xlabel('Dropout regularization parameter')
plt.title('Hyperparameter tuning for Supervisor network')
proxy = [plt.Rectangle((0,0),1,1, fc=pc.get_facecolor()[0]) for pc in cs.collections]
interval = (np.max(means) - np.min(means)) / 5

#interval = '%.2E' % Decimal(interval)

plt.legend(proxy, ['%.2E' % Decimal(np.min(means)) + "-" + '%.2E' % Decimal(np.min(means) + interval),
                   '%.2E' % Decimal(np.min(means) + interval) + "-" + '%.2E' % Decimal(np.min(means) + 2*interval),
                   '%.2E' % Decimal(np.min(means) + 2*interval) + "-" + '%.2E' % Decimal(np.min(means) + 3*interval),
                   '%.2E' % Decimal(np.min(means) + 3*interval) + "-" + '%.2E' % Decimal(np.min(means) + 4*interval),
                   '%.2E' % Decimal(np.min(means) + 4*interval) + "-" + '%.2E' % Decimal(np.min(means) + 5*interval)])
plt.show()

# Produce a scree plot comparable to PCA
plt.figure(dpi=800)
plt.plot(range(1,6), results_dropout_0_1[0])
plt.plot(range(1,6), results_dropout_0_2[0])
plt.plot(range(1,6), results_dropout_0_3[0])
plt.ylabel(r'Reconstruction error $\mathcal{L}_R$')
plt.xlabel(r'Latent dimension $\mathcal{H}$')
plt.legend(('Dropout = 0.1', 'Dropout = 0.2', 'Dropout = 0.3'))
plt.title(r'$\mathcal{L}_R$ over $\mathcal{H}$ for 1-layer autoencoder')
plt.show()

# Fit a XGboost classifier 
from xgboost import XGBRegressor, plot_importance
model = XGBRegressor()
model.fit(X, probs)
model.feature_importances_

# Plot the importance
plt.figure(dpi=800)
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.title('Feature importance based on XGboost')
plt.xlabel('Feature')
plt.ylabel('Importance score')
plt.show()

# Plot importance according to the package
plt.figure(dpi=800)
plot_importance(model)
plt.show()