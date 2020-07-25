from absl import app
from absl import flags

from dataloader.image_generators import train_val_test_image_generator
from eval_utils.visualization_utils import display_confusion_matrix
from ganomaly.ganomaly_eval import eval_scores

FLAGS = flags.FLAGS
flags.DEFINE_integer("crop_size", 128, "Shape of (S, S) to take from image")
flags.DEFINE_integer("latent_dim", 256, "Size of latent representation of model")
flags.DEFINE_integer("batch_size", 32, "Size of training batches")
flags.DEFINE_float("resize", 1.0, "Resizing factor for crops if necessary to fit in e.g. 64x64xc crops")
flags.DEFINE_string("ganomaly_checkpoint_dir", './ganomaly/128x128_256d/checkpoints',
                    "relative dir path to trained ganomaly checkpoint")
flags.DEFINE_string("data_path", None, "absolute dir path for image dataset with anomalies")
flags.DEFINE_float("score_threshold", 0.5, "Threshold for anomaly score predictions")


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

    predictions, labels = eval_scores(data_generator=test_img_gen,
                                      input_shape=input_shape,
                                      latent_dim=FLAGS.latent_dim,
                                      checkpoint_dir=FLAGS.ganomaly_checkpoint_dir,
                                      threshold=FLAGS.score_threshold)

    # plot confusion matrix for results
    display_confusion_matrix(predictions, labels)


if __name__ == '__main__':
    app.run(main)
