import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import ndimage
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape
from tensorflow.keras.models import Sequential

from eval.metric_visualizations import show_heatmap, show_ssim, show_triptych


class CAE(tf.keras.Model):

    def __init__(self, input_shape, latent_dim=128, log_dir='logs'):
        super(CAE, self).__init__()
        self.training_step = 0
        self._tb_image_writer = tf.summary.create_file_writer(log_dir + '/images')
        self._latent_dim = latent_dim
        self._input_shape = input_shape
        self.encoder = Sequential(
            [
                InputLayer(input_shape=self._input_shape),
                Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
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

        if K.learning_phase() and (self.training_step % 100 == 0):
            self.write_images_to_tb(inputs, decoded)

        return decoded

    def write_images_to_tb(self, inputs, decoded):
        print('step: ', self.training_step, ' - Writing images to TensorBoard...')
        with self._tb_image_writer.as_default():
            inputs = tf.cast(inputs, dtype=tf.uint8)
            decoded = tf.cast(decoded, dtype=tf.uint8)
            tf.summary.image("Original Images", inputs, max_outputs=4, step=self.training_step)
            tf.summary.image("Reconstructed Images", decoded, max_outputs=4, step=self.training_step)

    @tf.function
    def diff_map(self, inputs):
        reconstructed = self(inputs, training=False)
        diff_map = K.abs(tf.cast(reconstructed, tf.float64) - inputs)
        return diff_map, reconstructed

    def visualize_anomalies(self, inputs, method='heatmap', crop_size=128, labels=None):
        diff_map, reconstructed = self.diff_map(inputs)
        if method == 'triptych':
            show_triptych(inputs, reconstructed, diff_map, labels=labels, crop_size=crop_size)

        elif method == 'heatmap':
            show_heatmap(inputs, diff_map, labels=labels, crop_size=crop_size)

        elif method == 'ssim':
            show_ssim(inputs, reconstructed, labels=labels, crop_size=crop_size)

    def anomaly_scores(self, inputs, metric_fn):
        reconstructed = self(inputs)
        score = metric_fn(inputs, reconstructed)
        return score

    def detect_anomalies(self, image, min_threshold=25, percentile=99.5, min_area=10, label=None, debug=False,
                         crop_size=128):
        diff_map, _ = self.diff_map(image)
        diff_map = diff_map.numpy()[0].astype(np.uint8)

        grey_opening = ndimage.grey_opening(diff_map[:, :, 0], (3, 3), mode='nearest')
        grey_opening = cv2.medianBlur(grey_opening, 3)

        _, mask1 = cv2.threshold(grey_opening, min_threshold, 255, cv2.THRESH_BINARY)
        percentile = np.percentile(grey_opening, percentile)
        _, mask2 = cv2.threshold(grey_opening, percentile, 255, cv2.THRESH_BINARY)
        thresh_img = mask1 * mask2

        # try to connect close dots
        kernel = np.ones((3, 3))
        thresh_img - cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
        thresh_img - cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

        final_map = np.zeros_like(thresh_img)
        b = 5  # border from edge
        final_map[b:-b, b:-b] = thresh_img[b:-b, b:-b]  # get rid of edge differences (many FP)
        contours, hierarchy = cv2.findContours(final_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_area = [cv2.contourArea(contour) for contour in contours]
        # filter by contour area
        contours = [contour for contour, area in zip(contours, contours_area) if area >= min_area]

        if debug and label == 1:
            print('Contours area: ', contours_area)
            panels = np.zeros((crop_size, crop_size * 4, 1))
            panels[:, :crop_size, :] = image
            panels[:, crop_size:crop_size * 2, :] = diff_map
            panels[:, crop_size * 2:crop_size * 3, :] = np.expand_dims(grey_opening, -1)
            panels[:, crop_size * 3:crop_size * 4, :] = np.expand_dims(thresh_img * 255, -1)

            cv2.imshow('Image | Diff Map | Grey Open | Thresh | ',
                       panels.astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return contours


class CVAE(CAE):

    def __init__(self, input_shape, latent_dim=128, log_dir='logs'):
        super(CVAE, self).__init__(input_shape, latent_dim=latent_dim, log_dir=log_dir)
        self.encoder = Sequential(
            [
                InputLayer(input_shape=self._input_shape),
                Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                Flatten(),
                Dense(latent_dim + latent_dim),
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

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        logits = self.decoder(z)
        return logits
