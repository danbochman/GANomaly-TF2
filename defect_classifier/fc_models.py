import tensorflow as tf
from tensorflow.keras.layers import Dense


class LogisticRegressor(tf.keras.Model):

    def __init__(self, num_features=512):
        super(LogisticRegressor, self).__init__()
        self.dense = Dense(1, input_shape=(num_features,), activation='sigmoid')

    def call(self, inputs):
        logits = self.dense(inputs)
        return logits


class FullyConnected(tf.keras.Model):

    def __init__(self, num_features=512):
        super(FullyConnected, self).__init__()
        self.hidden = Dense(256, input_shape=(num_features,), activation='relu')
        self.final = Dense(1, activation='sigmoid')

    def call(self, inputs):
        h = self.hidden(inputs)
        logits = self.final(h)
        return logits
