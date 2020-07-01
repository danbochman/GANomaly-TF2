import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from dataloader.image_generators import train_val_image_generator
from models.autoencoders import CAE
from train.losses import mse_dssim_mixed_loss
from train.tensorboard_utils import XTensorBoard

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    crop_size = 128
    latent_dim = 64
    batch_size = 128

    image_data_path = "/media/jpowell/hdd/Data/AIS/RO2_OK_images/"
    # image_data_path = "/media/jpowell/hdd/Data/AIS/8C3W_per_Camera/"

    train_img_gen, val_img_gen = train_val_image_generator(image_data_path,
                                                           crop_size=crop_size,
                                                           batch_size=batch_size,
                                                           random_state=2,
                                                           preprocess=False)

    print('Initializing model...')

    cae = CAE(input_shape=(crop_size, crop_size, 1), latent_dim=latent_dim)
    cae.compile(optimizer='adam', loss=mse_dssim_mixed_loss)

    path_to_ae_model = 'RO2_BN_128x_64d_best_model.h5'
    if os.path.exists(path_to_ae_model):
        print('Loading model from checkpoint....')
        cae.load_weights(path_to_ae_model)

    callbacks = [XTensorBoard('./autoencoder/logs/'),
                 ModelCheckpoint(path_to_ae_model, monitor='val_loss', save_best_only=True, save_weights_only=False,
                                 verbose=1),
                 ReduceLROnPlateau(monitor='val_loss')]

    print('Training...')
    history = cae.fit(train_img_gen,
                      callbacks=callbacks,
                      steps_per_epoch=100,
                      epochs=200,
                      validation_data=val_img_gen,
                      validation_steps=20,
                      validation_freq=1)

    print('Finished training successfully')


if __name__ == '__main__':
    main()
