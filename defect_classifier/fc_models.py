import tensorflow as tf
from tensorflow.keras.layers import Dense


class LogisticRegressor(tf.keras.Model):

    def __init__(self, num_features=512):
        super(LogisticRegressor, self).__init__()
        self.dense = Dense(1, input_shape=(num_features,), activation='sigmoid')

    def call(self, inputs):
        logits = self.dense(inputs)
        return logits
