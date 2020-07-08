import tensorflow as tf
import tensorflow.keras.backend as K


def reconstruction_mse(original_img, reconstructed_img):
    mse = K.mean(K.square(original_img - reconstructed_img), axis=[1, 2, 3])
    return mse


def reconstruction_mse_loss(original_img, reconstructed_img):
    mse = K.mean(reconstruction_mse(original_img, reconstructed_img), axis=0)
    return mse


def dssim(original_img, reconstructed_img):
    ssim_score = tf.image.ssim(original_img, reconstructed_img, max_val=1.0, filter_size=11,
                               filter_sigma=1.5, k1=0.01, k2=0.03)
    return (1 - ssim_score) / 2


def mse_dssim_mixed(original_img, reconstructed_img):
    gamma = 10
    dssim_score = dssim(original_img, reconstructed_img)
    mse = reconstruction_mse(original_img, reconstructed_img)
    return mse - (gamma * dssim_score)


def mse_dssim_mixed_loss(original_img, reconstructed_img):
    mse_dssim = K.mean(mse_dssim_mixed(original_img, reconstructed_img), axis=0)
    return mse_dssim


def weighted_binary_crossentropy_loss(y_true, y_pred):
    eps = 1e-7
    anomaly_weight = 0.99
    wce = anomaly_weight * y_true * tf.math.log(eps + y_pred) + \
          (1.0 - anomaly_weight) * (1.0 - y_true) * tf.math.log(eps + 1.0 - y_pred)
    return -tf.reduce_mean(wce)


