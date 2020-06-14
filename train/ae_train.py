from tensorflow.keras.callbacks import TensorBoard

from dataloader.image_generator import image_generator
from models.autoencoders import CAE
from train.losses import reconstruction_mse


def main():
    img_gen = image_generator()
    img_sample = img_gen.__next__()
    input_shape = img_sample[0][0].shape

    cae = CAE(input_shape=input_shape)
    cae.compile(optimizer='adam',
                loss=reconstruction_mse)

    callbacks = [TensorBoard()]

    history = cae.fit_generator(img_gen,
                                callbacks=callbacks,
                                steps_per_epoch=24,
                                epochs=10)


if __name__ == '__main__':
    main()
