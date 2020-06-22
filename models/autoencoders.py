import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
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

    def visualize_anomalies(self, inputs, method='heatmap', labels=None):
        diff_map, reconstructed = self.diff_map(inputs)
        if method == 'triptych':
            self.show_triptych(inputs, reconstructed, diff_map, labels)

        elif method == 'heatmap':
            self.show_heatmap(inputs, diff_map, labels)

    def anomaly_scores(self, inputs, metric_fn):
        reconstructed = self(inputs)
        score = metric_fn(inputs, reconstructed)
        return score

    def detect_anomalies(self, inputs, min_threshold=50, percentile=99):
        diff_map, _ = self.diff_map(inputs)
        diff_map = diff_map.numpy()[0].astype(np.uint8)
        cv2.imshow('diff map', diff_map)

        diff_map = cv2.GaussianBlur(diff_map, (5, 5), 0)
        _, mask1 = cv2.threshold(diff_map, min_threshold, 255, cv2.THRESH_BINARY)
        percentile = np.percentile(diff_map, percentile)
        _, mask2 = cv2.threshold(diff_map, percentile, 255, cv2.THRESH_BINARY)
        thresh_img = mask1 * mask2
        cv2.imshow('mask1 ', mask1.astype(np.uint8))
        cv2.imshow('mask2 ', mask2.astype(np.uint8))

        thresh_img = thresh_img.astype(np.uint8) * 255
        cv2.imshow('thresh_img', thresh_img)

        kernel = np.ones((3, 3), np.uint8)
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('after opening', thresh_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

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
