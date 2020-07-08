import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, ReLU
from tensorflow.keras.layers import InputLayer, Flatten, Reshape, Input, Concatenate
from tensorflow.keras.models import Model, Sequential


class EGBAD(object):

    def __init__(self, input_shape, latent_dim=128):
        self._latent_dim = latent_dim
        self.Ex = Sequential(
            [
                InputLayer(input_shape=input_shape),
                Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same'),
                # LeakyReLU(),  # original BiGAN paper there's no activation, in EGBAD implementation there is
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same'),
                BatchNormalization(),
                LeakyReLU(0.1),
                Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same'),
                BatchNormalization(),
                LeakyReLU(0.1),
                Flatten(),
                Dense(latent_dim),
            ],
            name='Ex'
        )

        self._intermediate_shapes = self.infer_intermediate_shapes()

        self.Gz = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(units=1024),  # not sure why symmetry is broken here
                BatchNormalization(),
                ReLU(),
                Dense(units=self._intermediate_shapes[1]),
                BatchNormalization(),
                ReLU(),
                Reshape(target_shape=self._intermediate_shapes[0]),
                Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), padding='same'),
                BatchNormalization(),
                ReLU(),
                Conv2DTranspose(filters=1, kernel_size=4, strides=(2, 2), padding='same', activation='tanh'),
                #  I believe tanh is to preserve symmetry with input scaled to [-1, 1]
            ],
            name='Gz'
        )

        self.Dxz = Discriminator(input_shape, latent_dim).Dxz

    def infer_intermediate_shapes(self):
        flatten_layer = self.Ex.layers[-2]
        return flatten_layer.input_shape[1:], flatten_layer.output_shape[1:][0]


class Discriminator(object):
    def __init__(self, input_shape, latent_dim):

        self.Dx = tf.keras.Sequential(
            [
                InputLayer(input_shape=input_shape),
                Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='same'),
                LeakyReLU(0.1),
                # In EGBAD paper there's Dropout(0.5) here... not sure why... skipped
                Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='same'),
                BatchNormalization(),
                LeakyReLU(0.1),
                # Also here
                Flatten(),

            ],
            name='Dx'
        )

        self.Dz = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(512),
                LeakyReLU(0.1),
                # Dropout here
            ],
            name='Dz'
        )

        self.Dxz = self.discriminator(input_shape, latent_dim)

    def discriminator(self, input_shape, latent_dim):
        x = Input(shape=input_shape)
        z = Input(shape=(latent_dim,))
        dx = self.Dx(x)
        dz = self.Dz(z)
        dxdz = Concatenate(axis=1)([dx, dz])
        intermediate_layer = Dense(1024)(dxdz)
        intermediate_layer = LeakyReLU(0.1)(intermediate_layer)
        # Dropout(0.5)
        logits = Dense(1)(intermediate_layer)
        discriminator = Model(inputs=[x, z], outputs=[logits, intermediate_layer], name='Dxz')
        return discriminator
