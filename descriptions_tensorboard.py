def descr_auto_loss():
    return 'This graph shows the Mean-Squared Error (MSE) loss for the \
            Autoencoder network. As can be seen the test loss is consistenly \
            higher than the train loss as expected.'
            
def descr_auto_grads_embedder():
    return 'This graph shows the mean gradients in the bottom and top layer \
            of the embedder network. As can be seen the gradients decrease \
            over time.'

def descr_auto_grads_recovery():
    return 'This graph shows the mean gradients in the bottom and top layer \
            of the recovery network. As can be seen the gradients decrease \
            over time.'
            
def descr_supervisor_loss():
    return 'This graph shows the Mean-Squared Error (MSE) loss for the \
            Supervisor network. As can be seen the test loss is consistenly \
            higher than the train loss as expected.'
            
def descr_auto_grads_supervisor():
    return 'This graph shows the mean gradients in the bottom and top layer \
            of the supervisor network. As can be seen the gradients decrease \
            over time.'
            
def descr_auto_loss_joint_auto():
    return 'This graph shows the Mean-Squared Error (MSE) loss for the \
            Autoencoder network during the joint training phase. This loss is \
            a component used for the embedding and temporal learning phase of \
            TimeGAN.'

def descr_auto_loss_joint_supervisor():
    return 'This graph shows the Mean-Squared Error (MSE) loss for the \
            Supervisor network applied on the embedder during the joint \
            training phase. This loss is a component used for the embedding \
            and temporal learning phase of TimeGAN.'
            
def descr_supervisor_loss_joint():
    return 'This graph shows the Mean-Squared Error (MSE) loss for the \
            Supervisor network applied on the generator during the joint \
            training phase. This loss is a component used for temporal \
            learning phase of the generator in TimeGAN. (closed-loop training)'

def descr_generator_loss_joint():
    return 'This graph shows the Binary logistic cross-entropy loss for the \
            generator network during the joint training phase. The loss is a \
            component used for the unsupervised GAN learning phase of TimeGAN. \
            (open-loop training)'

def descr_discriminator_loss_joint():
    return 'This graph shows the Binary logistic cross-entropy loss for the \
            discriminator network during the joint training phase. The loss \
            is a component used for the unsupervised GAN learning phase of \
            TimeGAN. (open-loop training)'