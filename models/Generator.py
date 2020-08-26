from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense

# Generator network in Tensorflow 2.x
class Generator(Model):
    def __init__(self, tensorboard_folder_path, hparams, hidden_dim):
        super(Generator, self).__init__()
        self.LSTM1 = LSTM(units=10, 
                            return_sequences=True,
                            input_shape=(20,5),
                            kernel_initializer = 'he_uniform',
                            dropout = 0.2,
                            recurrent_dropout = 0,
                            name = 'LSTM1')
        self.LSTM2 = LSTM(units=7,
                            return_sequences=True,
                            kernel_initializer = 'he_uniform',
                            dropout = 0.2,
                            recurrent_dropout = 0,
                            name = 'LSTM2')
        self.LSTM3 = LSTM(units=hidden_dim,
                            return_sequences=True,
                            kernel_initializer = 'he_uniform',
                            dropout = 0.2,
                            recurrent_dropout = 0,
                            name = 'LSTM3')
        self.Dense1 = Dense(units=hidden_dim,
                            activation='sigmoid', # To ensure [0, 1]
                            name = 'Dense1')
        self.graph_has_been_written=False
        self.tensorboard_folder_path = tensorboard_folder_path

    def call(self, x, **kwargs): # Implement training = False when testing
        x = self.LSTM1(x)
        x = self.LSTM2(x)
        x = self.LSTM3(x)
        x = self.Dense1(x)
        return x