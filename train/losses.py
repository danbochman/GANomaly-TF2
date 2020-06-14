import tensorflow.keras.backend as K


def reconstruction_mse(original_img, reconstructed_img):
    mse = K.mean(K.square(original_img - reconstructed_img), axis=[1, 2, 3])
    return mse
