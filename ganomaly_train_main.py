from absl import app
from absl import flags

from dataloader.image_generators import train_val_test_image_generator
from ganomaly.ganomaly_trainer import train

FLAGS = flags.FLAGS
flags.DEFINE_integer("training_steps", 10001, "Number of training steps")
flags.DEFINE_integer("crop_size", 128, "Shape of (S, S) to take from image")
flags.DEFINE_integer("latent_dim", 256, "Size of latent representation of model")
flags.DEFINE_integer("batch_size", 32, "Size of training batches")
flags.DEFINE_float("lr", 0.0002, "Learning rate for optimizers")
flags.DEFINE_integer("generator_steps", 10, "how many steps should the generator train before switching")
flags.DEFINE_integer("discriminator_steps", 10, "how many steps should the discriminator train before switching")
flags.DEFINE_list("generator_loss_weights", [50, 1, 1], "weights for reconstruction, encoding, adversarial losses")
flags.DEFINE_boolean("augment", True, "whether to perform image augmentations on the image batch")


flags.DEFINE_integer("display_step", 50, "Writing frequency for TensorBoard")
flags.DEFINE_float("resize", 1.0, "Resizing factor for crops if necessary to fit in e.g. 64x64xc crops")
flags.DEFINE_integer("save_checkpoint_every_n_steps", 500, "Frequency for saving model checkpoints")
flags.DEFINE_string("data_path", None, "absolute dir path for (normal) image dataset")


def main(argv=None):
    # init data generator (64x64 or 32x32 size images are recommended - modify with crop_size & resize)
    train_img_gen, test_img_gen = train_val_test_image_generator(data_path=FLAGS.data_path,
                                                                 crop_size=FLAGS.crop_size,
                                                                 batch_size=FLAGS.batch_size,
                                                                 resize=FLAGS.resize,
                                                                 normalize=True,
                                                                 val_frac=0.0)

    # infer input shape from crop size and resizing factor
    s = int(FLAGS.crop_size * FLAGS.resize)
    input_shape = (s, s, 1)

    # recommended logs directory name for parallel experimentation with different crop sizes and latent dimensions
    SUGGESTED_LOGS_DIR_NAME = f'./ganomaly/{str(FLAGS.crop_size)}x{str(FLAGS.crop_size)}_{str(FLAGS.latent_dim)}d'

    # train on data generator
    train(data_generator=train_img_gen,
          input_shape=input_shape,
          training_steps=FLAGS.training_steps,
          latent_dim=FLAGS.latent_dim,
          lr=FLAGS.lr,
          generator_steps=FLAGS.generator_steps,
          discriminator_steps=FLAGS.discriminator_steps,
          gen_loss_weights=FLAGS.generator_loss_weights,
          do_augmentations=FLAGS.augment,
          display_step=FLAGS.display_step,
          save_checkpoint_every_n_steps=FLAGS.save_checkpoint_every_n_steps,
          logs_dir=SUGGESTED_LOGS_DIR_NAME)


if __name__ == '__main__':
    app.run(main)
