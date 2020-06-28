import os

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from dataloader.image_generators import train_val_image_generator
from models.autoencoders import CAE
from train.losses import mse_ssim_mixed

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class XTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self.model.training_step += 1


def main():
    crop_size = 256
    train_img_gen, val_img_gen = train_val_image_generator("/media/jpowell/hdd/Data/AIS/RO2_OK_images/",
                                                           crop_size=crop_size,
                                                           batch_size=48,
                                                           random_state=1,
                                                           preprocess=False)

    cae = CAE(input_shape=(crop_size, crop_size, 1), latent_dim=256)
    cae.compile(optimizer='adam',
                loss=mse_ssim_mixed)

    path_to_weights = 'weights/256d_best_weights.h5'
    callbacks = [XTensorBoard('logs'),
                 ModelCheckpoint(path_to_weights, monitor='val_loss', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss')]

    if os.path.exists(path_to_weights):
        cae.load_weights(path_to_weights)

    history = cae.fit(train_img_gen,
                      callbacks=callbacks,
                      steps_per_epoch=100,
                      epochs=100,
                      validation_data=val_img_gen,
                      validation_steps=100,
                      validation_freq=1)

    print(history)


if __name__ == '__main__':
    main()
