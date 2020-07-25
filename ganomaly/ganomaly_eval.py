import numpy as np
import tensorflow as tf
from tqdm import tqdm

from eval_utils.visualization_utils import show_histogram_and_pr_curve
from ganomaly.ganomaly_model import GANomaly

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


def eval_scores(data_generator,
                input_shape,
                latent_dim,
                checkpoint_dir,
                threshold):
    """
    Evaluation step for the GANomaly model. Initializes model and restores from checkpoint, loops over labeled test
    data, computes the encoding L2 loss for input data and normalizes scores to be from [0, 1].
    Displays PR curve for anomaly scores vs. labels.
    """

    # init GANomaly model
    ganomaly = GANomaly(input_shape=input_shape, latent_dim=latent_dim)
    enc_x = ganomaly.Ex
    dec_z = ganomaly.Gz
    enc_x_hat = ganomaly.Ex_hat

    # checkpoint writer
    checkpoint = tf.train.Checkpoint(enc_x=enc_x,
                                     dec_z=dec_z,
                                     enc_x_hat=enc_x_hat)
    # restore from checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    labels = []
    anomaly_scores = []
    for img_batch, label_batch in tqdm(data_generator):
        # encoder
        z = enc_x(img_batch, training=False)

        # decoder
        x_hat = dec_z(z, training=False)

        # 2nd encoder
        z_hat = enc_x_hat(x_hat, training=False)

        # encoder L1 distance (recommended by paper)
        anomaly_score = tf.norm(z - z_hat, ord=1, axis=1)

        labels.extend(label_batch)
        anomaly_scores.extend(anomaly_score)

    # visualize scores distributions and precision-recall
    show_histogram_and_pr_curve(anomaly_scores, labels)

    # assign predictions from anomaly scores and threshold
    predictions = (np.array(anomaly_scores) > threshold).astype(np.float)

    return predictions, labels