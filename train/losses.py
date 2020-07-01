import numpy as np
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


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)
