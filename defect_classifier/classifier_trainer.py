import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import Progbar

from defect_classifier.fc_models import LogisticRegressor, FullyConnected
from eval_utils.tensorboard_utils import figure_to_tf_image, confusion_matrix_figure
from ganomaly.ganomaly_model import GANomaly

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


class SSTrainer(object):
    def __init__(self,
                 train_generator,
                 val_generator,
                 input_shape,
                 latent_dim,
                 lr,
                 ganomaly_checkpoint_dir,
                 save_checkpoint_every_n_steps,
                 logs_dir):

        # data generators
        self.train_generator = train_generator
        self.val_generator = val_generator

        # init GANomaly model
        ganomaly = GANomaly(input_shape=input_shape, latent_dim=latent_dim)
        self.enc_x = ganomaly.Ex
        self.dec_z = ganomaly.Gz
        self.enc_x_hat = ganomaly.Ex_hat
        # load weights from checkpoint
        self.ganomaly_checkpoint = tf.train.Checkpoint(enc_x=self.enc_x,
                                                       dec_z=self.dec_z,
                                                       enc_x_hat=self.enc_x_hat)
        self.ganomaly_checkpoint.restore(tf.train.latest_checkpoint(ganomaly_checkpoint_dir))

        # init classifier
        self.classifier = FullyConnected(num_features=int(latent_dim * 2))

        # optimizers
        self.classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, name='cls_optimizer')

        # checkpoint writer
        self.logs_dir = logs_dir
        self.checkpoint_dir = self.logs_dir + '/checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.classifier_checkpoint = tf.train.Checkpoint(regressor=self.classifier,
                                                         reg_optimizer=self.classifier_optimizer)
        self.save_checkpoint_every_n_steps = save_checkpoint_every_n_steps

        # restore from checkpoint if exists
        self.classifier_checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        # summary writers
        self.train_writer = tf.summary.create_file_writer(logs_dir + '/train')
        self.val_writer = tf.summary.create_file_writer(logs_dir + '/validation')
        self.image_writer = tf.summary.create_file_writer(logs_dir + '/images')

    def train(self, training_steps=100, lr=0.00002, validation_freq=10, validation_steps=100, score_threshold=0.5):
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
        # progress bar
        progbar = Progbar(training_steps)

        for step in range(training_steps):
            progbar.update(step)

            # grab batch
            img_batch, label_batch = next(self.train_generator)
            label_batch = np.expand_dims(label_batch, -1).astype(np.float32)

            # GANomaly feed-forward
            z = self.enc_x(img_batch, training=False)

            # decoder
            x_hat = self.dec_z(z, training=False)

            # 2nd encoder
            z_hat = self.enc_x_hat(x_hat, training=False)

            # concat encodings
            encodings = tf.concat([z, z_hat], axis=1)

            with tf.GradientTape() as reg_tape:
                # input to logistic regressor
                logits = self.classifier(encodings)

                # compute loss
                train_loss = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(label_batch, logits, pos_weight=1000)
                )

            # compute gradients
            grad_gen = reg_tape.gradient(train_loss, self.classifier.trainable_variables)

            # apply gradients
            self.classifier_optimizer.apply_gradients(zip(grad_gen, self.classifier.trainable_variables))

            # save checkpoint
            if step % self.save_checkpoint_every_n_steps == 0:
                self.classifier_checkpoint.save(file_prefix=self.checkpoint_prefix)

            # write summaries
            if step % validation_freq == 0:
                with self.train_writer.as_default():
                    tf.summary.scalar("weighted crossentropy loss", train_loss, step=step)

                # validation
                print(f'   Step {step} | Training loss: {train_loss}')
                print('Starting eval phase...')
                self.eval(training_step=step, validation_steps=validation_steps, score_threshold=score_threshold)
                print('Resuming training...')


    def eval(self, training_step, validation_steps=100, score_threshold=0.5):
        labels = []
        losses = []
        scores = []
        for val_step in range(validation_steps):
            # grab batch
            img_batch, label_batch = next(self.val_generator)
            label_batch = np.expand_dims(label_batch, -1).astype(np.float32)
            labels.extend((label_batch))

            # GANomaly feed-forward
            z = self.enc_x(img_batch, training=False)

            # decoder
            x_hat = self.dec_z(z, training=False)

            # 2nd encoder
            z_hat = self.enc_x_hat(x_hat, training=False)

            # concat encodings
            encodings = tf.concat([z, z_hat], axis=1)

            # compute logits
            logits = self.classifier(encodings, training=False)
            scores.extend(logits)

            # compute loss
            validation_loss = tf.nn.weighted_cross_entropy_with_logits(label_batch, logits, pos_weight=100)
            losses.extend(validation_loss)

        # compute mean loss over all batches
        val_loss = tf.reduce_mean(losses)

        # write summaries to tensorboard
        with self.val_writer.as_default():
            tf.summary.scalar("weighted crossentropy loss", val_loss, step=training_step)

        with self.image_writer.as_default():
            predictions = tf.cast((np.array(scores) > score_threshold), tf.float32)

            # calculate the confusion matrix.
            cm = confusion_matrix(labels, predictions)
            # Log the confusion matrix as an image summary.
            cm_figure = confusion_matrix_figure(cm, class_names=['Normal', 'Anomaly'])
            cm_image = figure_to_tf_image(cm_figure)

            # Log the confusion matrix as an image summary.
            tf.summary.image("Confusion Matrix", cm_image, step=training_step)

        print(f'Step {training_step} | Validation loss: {val_loss}')