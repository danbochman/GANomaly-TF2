import os

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from dataloader.image_generators import train_image_generator
from models.autoencoders import CAE
from train.losses import reconstruction_mse, ssim_loss, mse_ssim_mixed

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    crop_size = 256
    train_img_gen = train_image_generator("/media/jpowell/hdd/Data/AIS/RO2_OK_images/", crop_size=crop_size,
                                          batch_size=64)

    cae = CAE(input_shape=(crop_size, crop_size, 1))
    cae.compile(optimizer='adam',
                loss=mse_ssim_mixed)

    path_to_weights = 'ssim_mse_mixed_best_weights.h5'
    callbacks = [TensorBoard('logs/scalars'),
                 ModelCheckpoint(path_to_weights, monitor='loss', save_best_only=True),
                 ReduceLROnPlateau(monitor='loss')]

    if os.path.exists(path_to_weights):
        cae.load_weights(path_to_weights)

    history = cae.fit(train_img_gen,
                      callbacks=callbacks,
                      steps_per_epoch=100,
                      epochs=100)


if __name__ == '__main__':
    main()
