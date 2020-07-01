import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.models import Sequential


class FilterSearcher(tf.keras.Model):

    def __init__(self, input_shape, frozen_autoencoder):
        super(FilterSearcher, self).__init__()
        self.training_step = 0
        self._autoencoder = frozen_autoencoder
        self._input_shape = input_shape
        self._tuner = Sequential(
            [
                InputLayer(input_shape=self._input_shape),
                Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(),
                AveragePooling2D(),
                Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(),
                AveragePooling2D(),
                Flatten(),
                Dense(32),
                BatchNormalization(),
                ReLU(),
                Dropout(0.25),
                Dense(1, activation='sigmoid'),

            ]
        )

    def call(self, inputs):
        diff_map, reconstructed = self._autoencoder.diff_map(inputs)
        prediction = self._tuner(diff_map)
        return prediction

