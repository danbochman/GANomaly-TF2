import os

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from dataloader.image_generators import train_image_generator
from models.autoencoders import CAE
from train.losses import reconstruction_mse


def main():
    crop_size = 256
    train_img_gen = train_image_generator("D:/Razor Labs/Projects/AIS/data/RO2/RO2_OK_images/", crop_size=crop_size)

    cae = CAE(input_shape=(crop_size, crop_size, 1))
    cae.compile(optimizer='adam',
                loss=reconstruction_mse)

    path_to_weights = 'best_weights.h5'
    callbacks = [TensorBoard(),
                 ModelCheckpoint(path_to_weights, monitor='loss', save_best_only=True),
                 ReduceLROnPlateau(monitor='loss')]

    if os.path.exists(path_to_weights):
        cae.load_weights(path_to_weights)

    history = cae.fit(train_img_gen,
                      callbacks=callbacks,
                      steps_per_epoch=100,
                      epochs=20)


if __name__ == '__main__':
    main()
