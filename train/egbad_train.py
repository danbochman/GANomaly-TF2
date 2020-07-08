import os

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from dataloader.image_generators import train_val_test_image_generator
from models.gans import EGBAD

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


def main():
    TRAINING_STEPS = 50001
    CROP_SIZE = 128
    LATENT_DIM = 64
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0002
    DISPLAY_STEP = 100
    RESIZE = 0.5
    LOG_DIR = './EGBAD/logs/'
    SAVE_EVERY_N_STEPS = 1000

    image_data_path = "/media/jpowell/hdd/Data/AIS/RO2_OK_images/"
    # image_data_path = "/media/jpowell/hdd/Data/AIS/8C3W_per_Camera/"

    train_img_gen, test_img_gen = train_val_test_image_generator(image_data_path,
                                                                 crop_size=CROP_SIZE,
                                                                 batch_size=BATCH_SIZE,
                                                                 resize=RESIZE,
                                                                 normalize=True,
                                                                 val_frac=0.0)

    input_shape = (int(CROP_SIZE * RESIZE), int(CROP_SIZE * RESIZE), 1)
    egbad = EGBAD(input_shape=input_shape, latent_dim=LATENT_DIM)

    gen = egbad.Gz
    enc = egbad.Ex
    dis = egbad.Dxz

    # optimizers
    optimizer_dis = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5, name='dis_optimizer')
    optimizer_gen = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5, name='gen_optimizer')
    optimizer_enc = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5, name='enc_optimizer')

    # summary writers
    scalar_writer = tf.summary.create_file_writer(LOG_DIR + 'scalars')
    image_writer = tf.summary.create_file_writer(LOG_DIR + 'images')

    # checkpoint writer
    checkpoint_dir = LOG_DIR + 'checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer_gen,
                                     discriminator_optimizer=optimizer_dis,
                                     encoder_optimizer=optimizer_enc,
                                     generator=gen,
                                     discriminator=dis,
                                     encoder=enc)

    # restore from checkpoint if exists
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    progres_bar = Progbar(TRAINING_STEPS)
    for step in range(TRAINING_STEPS):
        progres_bar.update(step)
        img_batch, label_batch = next(train_img_gen)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape, tf.GradientTape() as enc_tape:
            # encoder
            z_gen = enc(img_batch)

            # generator
            z = tf.random.normal((img_batch.shape[0], LATENT_DIM))
            x_gen = gen(z)
            x_rec = gen(z_gen)

            # discriminator
            logits_real, features_real = dis([img_batch, z_gen])
            logits_fake, features_fake = dis([x_gen, z])

            # losses
            # discriminator
            loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),
                                                                                  logits=logits_real))
            loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),
                                                                                  logits=logits_fake))
            loss_dis = loss_dis_gen + loss_dis_enc

            # generator
            loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake),
                                                                              logits=logits_fake))
            # encoder
            loss_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_real),
                                                                              logits=logits_real))

        # compute gradients
        grad_gen = gen_tape.gradient(loss_gen, gen.trainable_variables)
        grad_dis = dis_tape.gradient(loss_dis, dis.trainable_variables)
        grad_enc = enc_tape.gradient(loss_enc, enc.trainable_variables)

        # apply gradients
        optimizer_gen.apply_gradients(zip(grad_gen, gen.trainable_variables))
        optimizer_dis.apply_gradients(zip(grad_dis, dis.trainable_variables))
        optimizer_enc.apply_gradients(zip(grad_enc, enc.trainable_variables))

        # write summaries
        if step % DISPLAY_STEP == 0:
            with scalar_writer.as_default():
                # discriminator parts
                tf.summary.scalar("loss_discriminator", loss_dis, step=step)
                tf.summary.scalar("loss_dis_enc", loss_dis_enc, step=step)
                tf.summary.scalar("loss_dis_gen", loss_dis_gen, step=step)
                # generator
                tf.summary.scalar("loss_generator", loss_gen, step=step)
                # encoder
                tf.summary.scalar("loss_encoder", loss_enc, step=step)

            with image_writer.as_default():
                # [-1, 1] -> [0, 255]
                orig_display = tf.cast((img_batch + 1) * 127.5, tf.uint8)
                rec_display = tf.cast((x_rec + 1) * 127.5, tf.uint8)
                tf.summary.image('Original', orig_display, step=step, max_outputs=4)
                tf.summary.image('Reconstructed', rec_display, step=step, max_outputs=4)

        if step % SAVE_EVERY_N_STEPS == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()
