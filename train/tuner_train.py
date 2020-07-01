import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from dataloader.image_generators import test_image_generator
from models.autoencoders import CAE
from models.tuners import FilterSearcher
from train.losses import weighted_binary_crossentropy_loss
from train.tensorboard_utils import XTensorBoard

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    crop_size = 128
    latent_dim = 64
    batch_size = 128

    defect_data_path = "/media/jpowell/hdd/Data/AIS/RO2_NG_images/"

    test_img_gen = test_image_generator(defect_data_path, batch_size=batch_size, crop_size=crop_size, preprocess=False,
                                        repeat=True)

    print('Initializing autoencoder model...')
    input_shape = (crop_size, crop_size, 1)
    cae = CAE(input_shape=input_shape, latent_dim=latent_dim)
    path_to_ae_model = 'RO2_AC_128x_64d_best_model.h5'
    if os.path.exists(path_to_ae_model):
        print('Loading model from checkpoint....')
        cae.load_weights(path_to_ae_model)
        for layer in cae.layers:
            layer.trainable = False
        cae.compile()
        print('autoencoder loaded, frozen and compiled')


    path_to_tuner_model = 'RO2_Tuner.h5'
    callbacks = [XTensorBoard('./tuner/logs'),
                 ModelCheckpoint(path_to_tuner_model, monitor='loss', save_best_only=True, save_weights_only=False,
                                 verbose=1),
                 ReduceLROnPlateau(monitor='loss')]

    print('Initializing tuner model...')
    tuner_model = FilterSearcher(input_shape, cae)
    tuner_model.compile(optimizer='adam',
                        loss=weighted_binary_crossentropy_loss)

    print('Training...')
    history = tuner_model.fit(test_img_gen,
                              callbacks=callbacks,
                              steps_per_epoch=100,
                              epochs=100)

    print('Finished training successfully')


if __name__ == '__main__':
    main()
