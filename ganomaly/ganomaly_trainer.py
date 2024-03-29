import os

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from ganomaly.ganomaly_model import GANomaly
from dataloader.augmentations import augment

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


def train(data_generator,
          input_shape,
          training_steps,
          latent_dim,
          discriminator_steps,
          generator_steps,
          lr,
          gen_loss_weights,
          do_augmentations,
          display_step,
          save_checkpoint_every_n_steps,
          logs_dir):
    """
    Trainer function for the GANomaly model, all function parameters explanations are detailed in the
    ganomaly_train_main.py flags.
    The function flow is as follows:
    1. initializes the model, optimizers, tensorboard & checkpoints writers
    2. restores from checkpoint if exists
    3. Training loop:
        - train generator for k steps
        - train discriminator for j steps
        each forward pass looks like this:
            - encode img with E(x) -> get latent variable z
            - decode latent variable z with G(z) -> get reconstructed img x_hat
            - encode x_hat with E(x_hat) -> get latent variable z_hat
            - feed x, x_hat to discriminator
        compute losses:
            - reconstruction loss L1(x-x_hat)
            - encoding loss L2(z-z_hat)
            - adversarial loss L2(x_features-x_hat_features)
            - discriminator loss BCE

        - take optimizer steps for each submodel
        - write to tensorboard / save checkpoint every n steps
    """

    # init GANomaly model
    ganomaly = GANomaly(input_shape=input_shape, latent_dim=latent_dim)
    enc_x = ganomaly.Ex
    dec_z = ganomaly.Gz
    enc_x_hat = ganomaly.Ex_hat
    dis = ganomaly.Dx_x_hat

    # optimizers
    optimizer_dis = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5,
                                             name='dis_optimizer')  # SGD is also recommended for dis
    optimizer_gen = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, name='gen_optimizer')

    # summary writers
    scalar_writer = tf.summary.create_file_writer(logs_dir + '/scalars')
    image_writer = tf.summary.create_file_writer(logs_dir + '/images')

    # checkpoint writer
    checkpoint_dir = logs_dir + '/checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer_gen,
                                     discriminator_optimizer=optimizer_dis,
                                     enc_x=enc_x,
                                     dec_z=dec_z,
                                     enc_x_hat=enc_x_hat,
                                     dis=dis)

    # restore from checkpoint if exists
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    progres_bar = Progbar(training_steps)
    for step in range(training_steps):
        progres_bar.update(step)

        # generator phase
        for _ in range(generator_steps):

            # grab data from generator and (optional) augment
            img_batch, label_batch = next(data_generator)
            img_batch = tf.cast(img_batch, tf.float32)
            if do_augmentations:
                img_batch = augment(img_batch)

            with tf.GradientTape() as gen_tape:
                # encoder
                z = enc_x(img_batch)

                # decoder
                x_hat = dec_z(z)

                # 2nd encoder
                z_hat = enc_x_hat(x_hat)

                # discriminator
                logits_real, features_real = dis(img_batch)
                logits_fake, features_fake = dis(x_hat)

                # generator loss (label smoothing may be helpful)
                w_rec, w_enc, w_adv = gen_loss_weights
                loss_rec = tf.reduce_mean(tf.norm(img_batch - x_hat, ord=1, axis=(1, 2)))
                loss_enc = tf.reduce_mean(tf.norm(z - z_hat, ord=2, axis=1))
                loss_adv = tf.reduce_mean(tf.norm(features_real - features_fake, ord=2, axis=1))
                loss_gen = (w_rec * loss_rec) + (w_enc * loss_enc) + (w_adv * loss_adv)

                # compute gradients
                generator_vars = enc_x.trainable_variables + dec_z.trainable_variables + enc_x_hat.trainable_variables
                grad_gen = gen_tape.gradient(loss_gen, generator_vars)

                # apply gradients
                optimizer_gen.apply_gradients(zip(grad_gen, generator_vars))

        # discriminator phase
        for _ in range(discriminator_steps):

            # grab data from generator and (optional) augment
            img_batch, label_batch = next(data_generator)
            img_batch = tf.cast(img_batch, tf.float32)
            if do_augmentations:
                img_batch = augment(img_batch)

            with tf.GradientTape() as dis_tape:

                # encoder
                z = enc_x(img_batch)

                # decoder
                x_hat = dec_z(z)

                # discriminator
                logits_real, features_real = dis(img_batch)
                logits_fake, features_fake = dis(x_hat)

                # discriminator loss
                loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),
                                                                                   logits=logits_real))
                loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),
                                                                                   logits=logits_fake))
                loss_dis = (loss_real + loss_fake) * 0.5

                # compute gradients
                grad_dis = dis_tape.gradient(loss_dis, dis.trainable_variables)

                # apply gradients
                optimizer_dis.apply_gradients(zip(grad_dis, dis.trainable_variables))

        # write summaries
        if step % display_step == 0:
            with scalar_writer.as_default():
                tf.summary.scalar("loss_discriminator", loss_dis, step=step)
                tf.summary.scalar("loss_generator", loss_gen, step=step)
                tf.summary.scalar("loss_reconstruction", loss_rec, step=step)
                tf.summary.scalar("loss_encoding", loss_enc, step=step)
                tf.summary.scalar("loss_adversarial", loss_adv, step=step)

            with image_writer.as_default():
                # [-1, 1] -> [0, 255]
                orig_display = tf.cast((img_batch + 1) * 127.5, tf.uint8)
                rec_display = tf.cast((x_hat + 1) * 127.5, tf.uint8)
                concatenated_img = tf.concat([orig_display, rec_display], axis=2)
                tf.summary.image('Original | Reconstructed', concatenated_img, step=step, max_outputs=8)

        if step % save_checkpoint_every_n_steps == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
