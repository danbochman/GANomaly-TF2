import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape
from tensorflow.keras.models import Sequential

from dataloader.image_generator import image_generator
from train.losses import reconstruction_mse


class CAE(tf.keras.Model):

    def __init__(self, input_shape, latent_dim=128):
        super(CAE, self).__init__()
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

    def infer_inter_tensor_shape(self):
        flatten_layer = self.encoder.layers[-2]
        return flatten_layer.input_shape[1:], flatten_layer.output_shape[1:][0]

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    img_gen = image_generator()
    img_sample = img_gen.__next__()[0][0]
    input_shape = img_sample.shape

    cae = CAE(input_shape=input_shape,
              latent_dim=64)

    cae.compile(optimizer='adam',
                loss=reconstruction_mse)

    print('input shape: ', input_shape)
    print(cae.encoder.summary())
    print(cae.decoder.summary())
    print(cae.summary())
