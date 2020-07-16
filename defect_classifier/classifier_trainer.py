import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import Progbar

from defect_classifier.fc_models import LogisticRegressor
from eval_utils.tensorboard_utils import figure_to_tf_image, confusion_matrix_figure
from ganomaly.ganomaly_model import GANomaly

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


def train(train_generator,
          val_generator,
          input_shape,
          training_steps,
          latent_dim,
          lr,
          score_threshold,
          validation_freq,
          ganomaly_checkpoint_dir,
          save_checkpoint_every_n_steps,
          logs_dir):
    """
    Trainer function for the seconds stage following the GANomaly model, all function parameters explanations are detailed in the
    classifier_train_main.py flags.
    The function flow is as follows:
    1. initializes the model, optimizers, tensorboard & checkpoints writers
    2. restores from checkpoint if exists
    3. Training loop:
        - send img batch through trained GANomaly to get its outputs (encodings, reconstructions)
        - concat encodings to one feature vector (this is the training data for the classifier)
        - feed encodings to classifer and output logits
        compute losses:
            - binary cross-entropy
        - take optimizer step
        - write to tensorboard / save checkpoint every n steps
    """

    # init GANomaly model
    ganomaly = GANomaly(input_shape=input_shape, latent_dim=latent_dim)
    enc_x = ganomaly.Ex
    dec_z = ganomaly.Gz
    enc_x_hat = ganomaly.Ex_hat
    # load weights from checkpoint
    checkpoint = tf.train.Checkpoint(enc_x=enc_x,
                                     dec_z=dec_z,
                                     enc_x_hat=enc_x_hat)
    checkpoint.restore(tf.train.latest_checkpoint(ganomaly_checkpoint_dir))

    # init regressor
    regressor = LogisticRegressor(num_features=int(latent_dim * 2))

    # optimizers
    reg_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, name='reg_optimizer')

    # checkpoint writer
    checkpoint_dir = logs_dir + '/checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(regressor=regressor,
                                     reg_optimizer=reg_optimizer)

    # restore from checkpoint if exists
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # summary writers
    train_writer = tf.summary.create_file_writer(logs_dir + '/train')
    val_writer = tf.summary.create_file_writer(logs_dir + '/validation')
    image_writer = tf.summary.create_file_writer(logs_dir + '/images')

    progres_bar = Progbar(training_steps)
    for step in range(training_steps):
        progres_bar.update(step)

        # grab batch
        img_batch, label_batch = next(train_generator)
        label_batch = np.expand_dims(label_batch, -1).astype(np.float32)

        # GANomaly feed-forward
        z = enc_x(img_batch, training=False)

        # decoder
        x_hat = dec_z(z, training=False)

        # 2nd encoder
        z_hat = enc_x_hat(x_hat, training=False)

        # concat encodings
        encodings = tf.concat([z, z_hat], axis=1)

        with tf.GradientTape() as reg_tape:
            # input to logistic regressor
            logits = regressor(encodings)

            # compute loss
            train_loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(label_batch, logits, pos_weight=100)
            )

        # compute gradients
        grad_gen = reg_tape.gradient(train_loss, regressor.trainable_variables)

        # apply gradients
        reg_optimizer.apply_gradients(zip(grad_gen, regressor.trainable_variables))

        # write summaries
        if step % validation_freq == 0:
            with train_writer.as_default():
                tf.summary.scalar("weighted crossentropy loss", train_loss, step=step)

            # validation
            # grab batch
            img_batch, label_batch = next(val_generator)
            label_batch = np.expand_dims(label_batch, -1).astype(np.float32)

            # GANomaly feed-forward
            z = enc_x(img_batch, training=False)

            # decoder
            x_hat = dec_z(z, training=False)

            # 2nd encoder
            z_hat = enc_x_hat(x_hat, training=False)

            # concat encodings
            encodings = tf.concat([z, z_hat], axis=1)

            # compute logits
            logits = regressor(encodings, training=False)

            # compute loss
            validation_loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(label_batch, logits, pos_weight=100)
            )

            # write summaries to tensorboard
            with val_writer.as_default():
                tf.summary.scalar("weighted crossentropy loss", validation_loss, step=step)

            with image_writer.as_default():
                predictions = tf.cast((logits > score_threshold), tf.float32)

                # calculate the confusion matrix.
                cm = confusion_matrix(label_batch, predictions)
                # Log the confusion matrix as an image summary.
                cm_figure = confusion_matrix_figure(cm, class_names=['Normal', 'Anomaly'])
                cm_image = figure_to_tf_image(cm_figure)

                # Log the confusion matrix as an image summary.
                tf.summary.image("Confusion Matrix", cm_image, step=step)

        if step % save_checkpoint_every_n_steps == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
