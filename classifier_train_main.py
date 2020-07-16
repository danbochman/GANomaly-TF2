from absl import app
from absl import flags

from dataloader.image_generators import train_val_for_classifier
from defect_classifier.classifier_trainer import train

FLAGS = flags.FLAGS
flags.DEFINE_integer("training_steps", 501, "Number of training steps")
flags.DEFINE_integer("crop_size", 128, "Shape of (S, S) to take from image")
flags.DEFINE_integer("latent_dim", 256, "Size of latent representation of model")
flags.DEFINE_integer("batch_size", 128, "Size of training batches")
flags.DEFINE_float("lr", 0.0000001, "Learning rate for optimizers")
flags.DEFINE_float("score_threshold", 0.5, "score threshold for evaluating predictions")

flags.DEFINE_integer("validation_freq", 50, "Perform eval on validation set every n steps")
flags.DEFINE_float("resize", 1.0, "Resizing factor for crops if necessary to fit in e.g. 64x64xc crops")

flags.DEFINE_integer("save_checkpoint_every_n_steps", 100, "Frequency for saving model checkpoints")
flags.DEFINE_string("ganomaly_checkpoint_dir",
                    '/home/jpowell/PycharmProjects/AIS/ais_aae/ganomaly/128x128_256d/checkpoints',
                    "trained checkpoint for ganomaly model")
flags.DEFINE_string("logs_dir", './defect_classifier/logistic', "relative dir path to save TB events and checkpoints")
flags.DEFINE_string("data_path", "/media/jpowell/hdd/Data/AIS/RO2_NG_images/", "absolute dir path for image dataset")


def main(argv=None):
    # init train and validation data generators
    train_img_gen, val_img_gen = train_val_for_classifier(data_path=FLAGS.data_path,
                                                          crop_size=FLAGS.crop_size,
                                                          batch_size=FLAGS.batch_size,
                                                          resize=FLAGS.resize,
                                                          normalize=True,
                                                          val_frac=0.5)

    # infer input shape from crop size and resizing factor
    s = int(FLAGS.crop_size * FLAGS.resize)
    input_shape = (s, s, 1)

    # train on data generator
    train(train_generator=train_img_gen,
          val_generator=val_img_gen,
          input_shape=input_shape,
          training_steps=FLAGS.training_steps,
          latent_dim=FLAGS.latent_dim,
          lr=FLAGS.lr,
          score_threshold=FLAGS.score_threshold,
          validation_freq=FLAGS.validation_freq,
          save_checkpoint_every_n_steps=FLAGS.save_checkpoint_every_n_steps,
          ganomaly_checkpoint_dir=FLAGS.ganomaly_checkpoint_dir,
          logs_dir=FLAGS.logs_dir)


if __name__ == '__main__':
    app.run(main)
