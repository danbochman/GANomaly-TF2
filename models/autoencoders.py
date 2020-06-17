import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape
from tensorflow.keras.models import Sequential


class CAE(tf.keras.Model):

    def __init__(self, input_shape, latent_dim=128):
        super(CAE, self).__init__()
        tf.summary.experimental.set_step(0)
        self._latent_dim = latent_dim
        self._input_shape = input_shape
        self.encoder = Sequential(
            [
                InputLayer(input_shape=input_shape),
                Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same'),
                BatchNormalization(),
                LeakyReLU(),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same'),
                BatchNormalization(),
                LeakyReLU(),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same'),
                BatchNormalization(),
                LeakyReLU(),
                Flatten(),
                Dense(latent_dim),
            ]
        )

        self._inter_shape = self.infer_inter_tensor_shape()

        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(units=self._inter_shape[1]),
                BatchNormalization(),
                LeakyReLU(),
                Reshape(target_shape=self._inter_shape[0]),
                Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same'),
                BatchNormalization(),
                LeakyReLU(),
                Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same'),
                BatchNormalization(),
                LeakyReLU(),
                Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same'),
                BatchNormalization(),
                LeakyReLU(),
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
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        if K.learning_phase():
            self.tb_image_display(inputs, decoded)
        return decoded

    def tb_image_display(self, inputs, decoded):
        with self.tb_image_writer.as_default():
            inputs = tf.cast(inputs, tf.uint8)
            decoded = tf.cast(decoded, tf.uint8)
            summary_step = tf.summary.experimental.get_step()
            tf.summary.image("Original Images", inputs, max_outputs=4, step=summary_step)
            tf.summary.image("Reconstructed Images", decoded, max_outputs=4, step=summary_step)
            tf.summary.experimental.set_step(summary_step + 1)

    def diff_map(self, inputs):
        reconstructed = self(inputs)
        diff_map = K.abs(reconstructed - inputs)
        return diff_map, reconstructed

    def visualize_anomalies(self, inputs, mode='heatmap', labels=None):
        diff_map, reconstructed = self.diff_map(inputs)
        if mode == 'triptych':
            self.show_triptych(inputs, reconstructed, diff_map, labels)

        elif mode == 'heatmap':
            self.show_heatmap(inputs, diff_map, labels)

    @staticmethod
    def show_heatmap(inputs, diff_map, labels):
        for i in range(inputs.shape[0]):
            comparison = np.zeros((256, 512, 3))
            comparison[:, :256, :] = inputs[i]
            overlay = (inputs[i] * 0.7) + (diff_map[i] * 0.3)
            comparison[:, 256:512:, :] = cv2.applyColorMap(overlay.numpy().astype(np.uint8), cv2.COLORMAP_JET)
            label = 'Unknown'
            if labels is not None:
                label = str(labels[i])
            cv2.imshow(f'Original   |     Difference Heatmap     |     Label - {label}',
                       comparison.astype(np.uint8))
            cv2.waitKey(0)

    @staticmethod
    def show_triptych(inputs, reconstructed, diff_map, labels=None):
        for i in range(inputs.shape[0]):
            triptych = np.zeros((256, 256 * 3, 1))
            triptych[:, :256, :] = inputs[i]
            triptych[:, 256:512, :] = reconstructed[i]
            triptych[:, 512:768, :] = diff_map[i]
            label = 'Unknown'
            if labels is not None:
                label = str(labels[i])
            cv2.imshow(f'Original   |     Reconstructed    |     Difference      |     Label - {label}',
                       triptych.astype(np.uint8))
            cv2.waitKey(0)
