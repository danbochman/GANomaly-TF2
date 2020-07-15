import matplotlib.pyplot as plt
from absl import app
from absl import flags
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

from dataloader.image_generators import train_val_test_image_generator
from ganomaly.ganomaly_eval import eval_scores, eval_contours

FLAGS = flags.FLAGS
flags.DEFINE_integer("crop_size", 128, "Shape of (S, S) to take from image")
flags.DEFINE_integer("latent_dim", 256, "Size of latent representation of model")
flags.DEFINE_integer("batch_size", 32, "Size of training batches")
flags.DEFINE_float("resize", 1.0, "Resizing factor for crops if necessary to fit in e.g. 64x64xc crops")
flags.DEFINE_string("logs_dir", './ganomaly/128x128_256d', "relative dir path to save TB events and checkpoints")
flags.DEFINE_string("data_path", "/media/jpowell/hdd/Data/AIS/RO2_NG_images/", "absolute dir path for image dataset")
flags.DEFINE_string("method", 'contours', "method for evaluating the ganomaly model")

# FLAGS for anomaly score method
flags.DEFINE_boolean("display", True, "Show histograms and pr curve")
flags.DEFINE_float("score_threshold", 0.2, "Threshold for anomaly score predictions")

# FLAGS for contour method
flags.DEFINE_boolean("debug", False, "Visualize anomaly detection pipeline for contour method")
flags.DEFINE_boolean("show_contours", False, "Display polygons around contours if detected")
flags.DEFINE_integer("diff_threshold", 25, "absolute difference threshold for difference map")
flags.DEFINE_float("min_percentile", 99.5, "minimum percentile value for difference map threshold")
flags.DEFINE_integer("min_area", 10, "minimum area for contour (in pixels) to be considered an anomaly")


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

    if FLAGS.method == 'scores':
        predictions, labels = eval_scores(data_generator=test_img_gen,
                                          input_shape=input_shape,
                                          latent_dim=FLAGS.latent_dim,
                                          logs_dir=FLAGS.logs_dir,
                                          display=FLAGS.display,
                                          threshold=FLAGS.score_threshold)

    if FLAGS.method == 'contours':
        predictions, labels = eval_contours(data_generator=test_img_gen,
                                            input_shape=input_shape,
                                            latent_dim=FLAGS.latent_dim,
                                            logs_dir=FLAGS.logs_dir,
                                            show_contours=FLAGS.show_contours,
                                            debug=FLAGS.debug,
                                            threshold=FLAGS.diff_threshold,
                                            min_percentile=FLAGS.min_percentile,
                                            min_area=FLAGS.min_area)

    cm = confusion_matrix(labels, predictions)
    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly']).plot()
    plt.show()


if __name__ == '__main__':
    app.run(main)
