from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense

# Supervisor network in Tensorflow 2.x
class Supervisor(Model):
    def __init__(self, tensorboard_folder_path, hparams):
        super(Supervisor, self).__init__()
        self.LSTM1 = LSTM(units = 7,
                          return_sequences=True,
                          kernel_initializer = 'he_uniform',
                          dropout = 0.2,
                          recurrent_dropout = 0.2,
                          input_shape=(20,4),
                          name = 'LSTM1')
        self.LSTM2 = LSTM(units=6,
                          return_sequences=True,
                          kernel_initializer = 'he_uniform',
                          dropout = 0.2,
                          recurrent_dropout = 0.2,
                          name='LSTM2')
        self.Dense1 = Dense(units=4,
                            activation='sigmoid', # To stay in the [0,1] range
                            name = 'Dense1')
        self.graph_has_been_written=False
        self.tensorboard_folder_path=tensorboard_folder_path
        
    def call(self, x, **kwargs):
        x = self.LSTM1(x)
        x = self.LSTM2(x)
        x = self.Dense1(x)
        
        # # Print the graph in TensorBoard
        # if not self.graph_has_been_written:
        #     model_graph = x.graph
        #     writer = tf.compat.v1.summary.FileWriter(logdir=self.tensorboard_folder_path,
        #                                              graph=model_graph)
        #     writer.flush()
        #     self.graph_has_been_written = True
        #     print("Wrote eager graph to:", self.tensorboard_folder_path)
       
        return x