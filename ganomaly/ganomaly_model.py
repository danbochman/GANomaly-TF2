import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Dropout
from tensorflow.keras.layers import InputLayer, Flatten, Reshape, Input
from tensorflow.keras.models import Model, Sequential


class GANomaly(object):
    """
    Regular Python class meant to be used as a convenient way to initialize all the moving parts and submodels necessary
    for running a GANomaly architecture [https://arxiv.org/pdf/1805.06725v3.pdf]
    The model architecture is adaptive to the input_shape but (128, 128, c), (64, 64, c) or (32, 32, c) are recommended
    for optimal results.
    """

    def __init__(self, input_shape=(128, 128, 3), latent_dim=128):
        self._k_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)  # quite important
        self._latent_dim = latent_dim
        self._input_shape = input_shape

        # Encoder(x) model
        self.Ex = self.make_encoder(name='Ex')

        self._intermediate_shapes = self.infer_intermediate_shapes()  # to preserve symmetry

        # Generator(z) / Decoder(z) model
        self.Gz = self.make_decoder(name='Gz')

        # Encoder(x_hat) model
        self.Ex_hat = self.make_encoder(name='Ex_hat')

        # Discriminator(x, x_hat) model
        self.Dx_x_hat = self.make_discriminator('Dx_x_hat')

    def infer_intermediate_shapes(self):
        """
        Used as to infer the intermediate shapes needed for the generator model to act as an inverse to the encoder
        model
        :return tuple: (final feature map shape before flatten, after flatten)
        """
        flatten_layer = self.Ex.layers[-2]
        return flatten_layer.input_shape[1:], flatten_layer.output_shape[1:][0]

    def make_encoder(self, name='Ex'):
        model = Sequential(
            [
                InputLayer(input_shape=self._input_shape),
                Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer=self._k_init),
                LeakyReLU(0.1),
                Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='same', kernel_initializer=self._k_init),
                BatchNormalization(),
                LeakyReLU(0.1),
                Conv2D(filters=128, kernel_size=4, strides=(2, 2), padding='same', kernel_initializer=self._k_init),
                BatchNormalization(),
                LeakyReLU(0.1),
                Flatten(),
                Dense(self._latent_dim, kernel_initializer=self._k_init),
            ],
            name=name
        )
        return model

    def make_decoder(self, name='Gz'):
        model = tf.keras.Sequential(
            [
                InputLayer(input_shape=(self._latent_dim,)),
                Dense(units=self._intermediate_shapes[1], kernel_initializer=self._k_init),
                BatchNormalization(),
                ReLU(),
                Reshape(target_shape=self._intermediate_shapes[0]),
                Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2), padding='same',
                                kernel_initializer=self._k_init),
                BatchNormalization(),
                ReLU(),
                Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), padding='same',
                                kernel_initializer=self._k_init),
                BatchNormalization(),
                ReLU(),
                Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same',
                                kernel_initializer=self._k_init,
                                activation='tanh'),
                #  I believe tanh is to preserve symmetry with input scaled to [-1, 1]
            ],
            name=name
        )
        return model

    def make_discriminator(self, name='Discriminator'):
        # standard DCGAN discriminator
        inputs = Input(shape=self._input_shape)
        x = Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same', kernel_initializer=self._k_init)(inputs)
        x = LeakyReLU(0.1)(x)
        x = Dropout(0.3)(x)
        x = Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same', kernel_initializer=self._k_init)(x)
        x = LeakyReLU(0.1)(x)

        # we will also output the features for feature matching loss
        features = Flatten()(x)
        logits = Dense(1, kernel_initializer=self._k_init)(features)
        discriminator = Model(inputs=inputs, outputs=[logits, features], name=name)
        return discriminator
