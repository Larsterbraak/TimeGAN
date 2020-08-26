import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

# Discriminator network for GAN in latent space in Tensorflow 2.x
class Discriminator(Model):
    def __init__(self, tensorboard_folder_path, hparams, hidden_dim):
        super(Discriminator, self).__init__()
        self.LSTM1 = Bidirectional(LSTM(units=7,
                                        return_sequences=True,
                                        kernel_initializer = 'he_uniform',
                                        dropout = 0.2,
                                        recurrent_dropout = 0,
                                        input_shape=(20,hidden_dim),
                                        name='LSTM1'))
        self.LSTM2 = Bidirectional(LSTM(units=4,
                                        return_sequences=True,
                                        kernel_initializer = 'he_uniform',
                                        dropout = 0.2,
                                        recurrent_dropout = 0,
                                        name='LSTM2'))
        self.LSTM3 = Bidirectional(LSTM(units=2,
                                        return_sequences=True,
                                        kernel_initializer = 'he_uniform',
                                        dropout = 0.2,
                                        recurrent_dropout = 0,
                                        name='LSTM3'))
        self.Dense1 = Dense(units=1,
                            activation=None,
                            name='Dense1')
        self.graph_has_been_written=False
        self.i = 0
        self.tensorboard_folder_path = tensorboard_folder_path
        
    def call(self, x, **kwargs): # Implement training = False when testing 
        x = self.LSTM1(x)
        x = self.LSTM2(x)
        x = self.LSTM3(x)
        x = self.Dense1(x)
        
        # Print the graph in TensorBoard
        if not self.graph_has_been_written and self.i == 0:
            model_graph = x.graph
            writer = tf.compat.v1.summary.FileWriter(logdir=self.tensorboard_folder_path,
                                                     graph=model_graph)
            writer.flush()
            self.graph_has_been_written = True
            print("Wrote eager graph to:", self.tensorboard_folder_path)
        
        self.i = self.i + 1 # Log the number of calls to the discriminator model 
        return x
    
    def predict(self, x):
        x = self.LSTM1(x) # Perform the normal training steps
        x = self.LSTM2(x)
        x = self.LSTM3(x)
        x = self.Dense1(x)
        
        x = tf.math.sigmoid(x)        
        return x
        
        