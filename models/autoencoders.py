import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape
from tensorflow.keras.models import Sequential


class CAE(tf.keras.Model):

    def __init__(self, input_shape, latent_dim=128):
        super(CAE, self).__init__()
        self._train_step = 0
        self._latent_dim = latent_dim
        self._input_shape = input_shape
        self.encoder = Sequential(
            [
                InputLayer(input_shape=input_shape),
                Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
                Flatten(),
                Dense(latent_dim),
            ]
        )

        self._inter_shape = self.infer_inter_tensor_shape()

        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(units=self._inter_shape[1], activation='relu'),
                Reshape(target_shape=self._inter_shape[0]),
                Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                # No activation
                Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )
        self.tb_image_writer = tf.summary.create_file_writer('logs/images')

    def infer_inter_tensor_shape(self):
        flatten_layer = self.encoder.layers[-2]
        return flatten_layer.input_shape[1:], flatten_layer.output_shape[1:][0]

    def call(self, inputs):
        self._train_step += 1
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        if K.learning_phase():
            self.tb_image_display(inputs, decoded)
        return decoded

    def tb_image_display(self, inputs, decoded):
        with self.tb_image_writer.as_default():
            inputs = tf.cast(inputs, tf.uint8)
            decoded = tf.cast(decoded, tf.uint8)
            tf.summary.image("Original Images", inputs, max_outputs=1, step=self._train_step)
            tf.summary.image("Reconstructed Images", decoded, max_outputs=1, step=self._train_step)

        return decoded
