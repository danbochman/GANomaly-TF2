import cv2
import numpy as np
from scipy import ndimage

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential

from eval.metric_visualizations import show_lpf, show_heatmap, show_ssim, show_triptych
from train.losses import numpy_mse_ssim_mixed, mse_ssim_mixed

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

    def infer_inter_tensor_shape(self):
        flatten_layer = self.encoder.layers[-2]
        return flatten_layer.input_shape[1:], flatten_layer.output_shape[1:][0]

    def call(self, inputs):
        # self._in = inputs
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        # self._out = decoded
        return decoded


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

    def detect_anomalies(self, image, min_threshold=20, percentile=99, mean_loss=175, label=None, debug=False):
        diff_map, _ = self.diff_map(image)
        diff_map = diff_map.numpy()[0].astype(np.uint8)
        reconstruction_loss = self.anomaly_scores(tf.convert_to_tensor(image, dtype=tf.float32), mse_ssim_mixed)

        class_error = False
        if debug and label:
            err_cond_1 = (reconstruction_loss > mean_loss) and (label == 0)
            err_cond_2 = (reconstruction_loss <= mean_loss) and (label == 1)
            class_error = err_cond_1 or err_cond_2

        if reconstruction_loss >= mean_loss:
            # blurred = cv2.medianBlur(diff_map, 3)
            grey_opening = ndimage.grey_opening(diff_map[:, :, 0], (5, 5), mode='reflect')

            _, mask1 = cv2.threshold(grey_opening, min_threshold, 255, cv2.THRESH_BINARY)
            percentile = np.percentile(grey_opening, percentile)
            _, mask2 = cv2.threshold(grey_opening, percentile, 255, cv2.THRESH_BINARY)
            thresh_img = mask1 * mask2

            final_map = np.zeros_like(thresh_img)
            b = 15  # border from edge
            final_map[b:-b, b:-b] = thresh_img[b:-b, b:-b]   # get rid of edge differences (many FP)
            contours, hierarchy = cv2.findContours(final_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if debug:
                print(reconstruction_loss)
                panels = np.zeros((256, 256 * 5, 4))
                panels[:, :256, :] = image
                panels[:, 256:512, :] = diff_map
                panels[:, 512:768, :] = np.expand_dims(grey_opening, -1)
                panels[:, 768:1024, :] = np.expand_dims(thresh_img * 255, -1)

                cv2.imshow('Image | Diff Map | Grey Open | Thresh | ',
                           panels.astype(np.uint8))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return contours

        else:
            return []
