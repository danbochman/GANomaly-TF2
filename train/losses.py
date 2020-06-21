import tensorflow.keras.backend as K
import tensorflow as tf

def reconstruction_mse(original_img, reconstructed_img):
    mse = K.mean(K.square(original_img - reconstructed_img), axis=[1, 2, 3])
    return mse


def ssim_loss(original_img, reconstructed_img):
    ssim_score = tf.image.ssim(original_img, reconstructed_img, max_val=1.0, filter_size=11,
                      filter_sigma=1.5, k1=0.01, k2=0.03)
    return 1-ssim_score


def mse_ssim_mixed(original_img, reconstructed_img):
    gamma = 100
    ssim_score = tf.image.ssim(original_img, reconstructed_img, max_val=1.0, filter_size=11,
                      filter_sigma=1.5, k1=0.01, k2=0.03)
    mse = K.mean(K.square(original_img - reconstructed_img), axis=[1, 2, 3])

    return mse - (gamma * ssim_score)