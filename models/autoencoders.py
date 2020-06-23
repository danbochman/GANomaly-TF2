import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential

from eval.metric_visualizations import show_lpf, show_heatmap, show_ssim, show_triptych


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
            show_triptych(inputs, reconstructed, diff_map, labels)

        elif method == 'heatmap':
            show_heatmap(inputs, diff_map, labels)

        elif method == 'ssim':
            show_ssim(inputs, reconstructed, labels)

        elif method == 'lpf':
            show_lpf(inputs, diff_map, labels)

    def anomaly_scores(self, inputs, metric_fn):
        reconstructed = self(inputs)
        score = metric_fn(inputs, reconstructed)
        return score

    def detect_anomalies(self, image, min_threshold=20, percentile=99, display=False):
        diff_map, _ = self.diff_map(image)
        diff_map = diff_map.numpy()[0].astype(np.uint8)

        _, mask1 = cv2.threshold(diff_map, min_threshold, 255, cv2.THRESH_BINARY)
        percentile = np.percentile(diff_map, percentile)
        _, mask2 = cv2.threshold(diff_map, percentile, 255, cv2.THRESH_BINARY)
        thresh_img = mask1 * mask2

        kernel = np.ones((5, 5), np.uint8)
        thresh_img_opened = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
        thresh_img_closed = cv2.morphologyEx(thresh_img_opened, cv2.MORPH_CLOSE, kernel)

        thresh_img_closed = thresh_img_closed.astype(np.uint8)
        final_map = np.zeros_like(thresh_img_closed)
        b = 5  # border from edge
        final_map[b:-b, b:-b] = thresh_img_closed[b:-b, b:-b]   # get rid of edge differences (many FP)
        contours, hierarchy = cv2.findContours(final_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if display:
            panels = np.zeros((256, 256 * 7, 1))
            panels[:, :256, :] = image
            panels[:, 256:512, :] = diff_map
            panels[:, 512:768, :] = np.expand_dims(mask1.astype(np.uint8), -1)
            panels[:, 768:1024, :] = np.expand_dims(mask2.astype(np.uint8), -1)
            panels[:, 1024:1280, :] = np.expand_dims(thresh_img * 255, -1)
            panels[:, 1280:1536, :] = np.expand_dims(thresh_img_opened * 255, -1)
            panels[:, 1536:1792, :] = np.expand_dims(final_map * 255, -1)

            cv2.imshow('Image | Diff Map | Mask 1 | Mask 2 | Before Opening | After Opening  | Final Map',
                       panels.astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return contours
