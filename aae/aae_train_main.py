from absl import app
from absl import flags

from aae.aae_trainer import train
from dataloader.image_generators import train_val_test_image_generator

FLAGS = flags.FLAGS
flags.DEFINE_integer("training_steps", 1001, "Number of training steps")
flags.DEFINE_integer("crop_size", 160, "Shape of (S, S) to take from image | Recommended: 128, 160, 200, 256")
flags.DEFINE_integer("latent_dim", 200, "Size of latent representation of model")
flags.DEFINE_integer("batch_size", 32, "Size of training batches")
flags.DEFINE_float("lr", 0.0002, "Learning rate for optimizers")
flags.DEFINE_integer("generator_steps", 10, "how many steps should the generator train before switching")
flags.DEFINE_integer("discriminator_steps", 10, "how many steps should the discriminator train before switching")
flags.DEFINE_list("generator_loss_weights", [50, 1], "weights for reconstruction, encoding, adversarial losses")
flags.DEFINE_boolean("augment", True, "Whether images will be augmented during training")

flags.DEFINE_integer("display_step", 10, "Writing frequency for TensorBoard")
flags.DEFINE_float("resize", 1.0, "Resizing factor for crops if necessary to fit in e.g. 64x64xc crops")
flags.DEFINE_integer("save_checkpoint_every_n_steps", 100, "Frequency for saving model checkpoints")
flags.DEFINE_string("data_path", "/media/jpowell/hdd/Data/AIS/RO2_OK_images/", "absolute dir path for image dataset")


def main(argv=None):
    # init data generator
    train_img_gen, test_img_gen = train_val_test_image_generator(data_path=FLAGS.data_path,
                                                                 crop_size=FLAGS.crop_size,
                                                                 batch_size=FLAGS.batch_size,
                                                                 resize=FLAGS.resize,
                                                                 normalize=True,
                                                                 val_frac=0.0)

    # infer input shape from crop size and resizing factor
    s = int(FLAGS.crop_size * FLAGS.resize)
    input_shape = (s, s, 1)

    # train on data generator
    SUGGESTED_LOGS_DIR_NAME = f'{str(FLAGS.crop_size)}x{str(FLAGS.crop_size)}_{str(FLAGS.latent_dim)}d'
    train(data_generator=train_img_gen,
          input_shape=input_shape,
          training_steps=FLAGS.training_steps,
          latent_dim=FLAGS.latent_dim,
          lr=FLAGS.lr,
          do_augmentations=FLAGS.augment,
          generator_steps=FLAGS.generator_steps,
          discriminator_steps=FLAGS.discriminator_steps,
          gen_loss_weights=FLAGS.generator_loss_weights,
          display_step=FLAGS.display_step,
          save_checkpoint_every_n_steps=FLAGS.save_checkpoint_every_n_steps,
          logs_dir=SUGGESTED_LOGS_DIR_NAME)


if __name__ == '__main__':
    app.run(main)
