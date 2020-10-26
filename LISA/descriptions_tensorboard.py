"""
MSc Thesis Quantitative Finance - Erasmus University Rotterdam
Title: Interest rate risk simulation using TimeGAN after the EONIA-ESTER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updates: Sep 16th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Inputs
N.A.
Outputs
(1) Descriptions for TF 2.0 Tensorboard visualizations of TimeGAN training
"""

def descr_pretrain_auto_loss():
    return 'This graph shows the Mean-Squared Error (MSE) loss for the \
            Autoencoder network for the training and test dataset after \
            every training iteration in the pre-training phase.'
            
def descr_pretrain_auto_embedder_grads():
    return 'This graph shows the sample mean of the gradient norm in \
            the bottom and top layer of the embedder network for every \
            minibatch training iteration in the pre-training phase.'

def descr_pretrain_auto_recovery_grads():
    return 'This graph shows the sample mean of the gradient norm in \
            the bottom and top layer of the recovery network for every \
            minibatch training iteration in the pre-training phase.'
            
def descr_pretrain_supervisor_loss():
    return 'This graph shows the Mean-Squared Error (MSE) loss for the \
            Supervisor network for the training and test dataset after \
            every training iteration in the pre-training phase.'
            
def descr_pretrain_supervisor_grads():
    return 'This graph shows the sample mean of the gradient norm in \
            the bottom and top layer of the Supervisor network for every \
            minibatch training iteration in the pre-training phase.'
            
def descr_joint_auto_loss():
    return 'This graph shows the Mean-Squared Error (MSE) loss for the \
            Autoencoder network for the training and test dataset after \
            every training iteration in the joint training phase.'

def descr_joint_supervisor_loss():
    return 'This graph shows the Mean-Squared Error (MSE) loss for the \
            Supervisor network for the training and test dataset after \
            every training iteration in the joint training phase.'

def descr_joint_generator_loss_wasserstein():
    return 'This graph shows the Wasserstein loss for the Generator \
            network for the training and test dataset after every training \
            iteration in the joint training phase.'

def descr_joint_generator_loss():
    return 'This graph shows the negative cross-entropy (CE) loss for the \
            Generator network for the training and test dataset after \
            every training iteration in the joint training phase.'

def descr_joint_generator_grads():
    return 'This graph shows the sample mean of the gradient norm in \
            the bottom and top layer of the Generator network for every \
            minibatch training iteration in joint training phase.'

def descr_joint_discriminator_loss_wasserstein():
    return 'This graph shows the Wasserstein loss for the Discriminator \
            network for the training and test data after every training \
            iteration in the joint training phase.'

def descr_joint_discriminator_loss():
    return 'This graph shows the negative cross-entropy (CE) loss for the \
            Discriminator network for the training and test dataset after \
            every training iteration in the joint training phase.'

def descr_joint_discriminator_grads():
    return 'This graph shows the sample mean of the gradient norm in \
            the bottom and top layer of the Discriminator network for every \
            minibatch training iteration in the joint training phase.'

def descr_joint_accuracy():
    return 'Discriminator accuracy of fake and real data samples after \
            every training iteration in the joint training phase.'
            
def descr_joint_feature_matching_loss():
    return 'This graph shows the sample mean of the Feature Matching (FM) \
            loss for the Generator network after every training iteration \
            in the joint training phase.'

def descr_joint_gradient_penalty():
    return 'This graph shows the sample mean of the gradient penalty for \
            the Discriminator network after every minibatch training \
            iterations in the joint training phase.'
