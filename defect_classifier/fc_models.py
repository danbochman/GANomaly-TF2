import tensorflow as tf
from tensorflow.keras.layers import Dense

class Linear(tf.keras.Model):

    def __init__(self, num_features=512):
        super(Linear, self).__init__()
        self.dense = Dense(1, input_shape=num_features, kernel_regularizer='l2')

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)