import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage
from tqdm import tqdm

from eval_utils.visualization_utils import show_histogram_and_pr_curve
from aae.aae_model import AAE

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


def eval_scores(data_generator,
                input_shape,
                latent_dim,
                logs_dir,
                display,
                threshold):
    """
    Evaluation step for the AAE model. Initializes model and restores from checkpoint, loops over labeled test
    data, computes the reconstruction L2 loss for input data and normalizes scores to be from [0, 1].
    Displays PR curve for anomaly scores vs. labels.
    """

    # init GANomaly model
    aae = AAE(input_shape=input_shape, latent_dim=latent_dim)
    enc_x = aae.Ex
    dec_z = aae.Gz

    # checkpoint writer
    checkpoint_dir = logs_dir + '/checkpoints'
    checkpoint = tf.train.Checkpoint(enc_x=enc_x,
                                     dec_z=dec_z)
    # restore from checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    labels = []
    anomaly_scores = []
    for img_batch, label_batch in tqdm(data_generator):
        # encoder
        z = enc_x(img_batch, training=False)

        # decoder
        x_hat = dec_z(z, training=False)

        # reconstruction L1 distance
        anomaly_score = tf.norm(img_batch - x_hat, ord=1, axis=1)

        labels.extend(label_batch)
        anomaly_scores.extend(anomaly_score)

    if display:
        show_histogram_and_pr_curve(anomaly_scores, labels)

    # assign predictions from anomaly scores and threshold
    predictions = (anomaly_scores > threshold).astype(np.float)

    return predictions, labels


def eval_contours(data_generator,
                  input_shape,
                  latent_dim,
                  logs_dir,
                  show_contours,
                  debug,
                  threshold,
                  min_percentile,
                  min_area):
    """
    Evaluation step for the AAE model. Initializes model and restores from checkpoint, loops over labeled test
    data, finds contours from difference map (x - x_hat)
    """

    # init GANomaly model
    aae = AAE(input_shape=input_shape, latent_dim=latent_dim)
    enc_x = aae.Ex
    dec_z = aae.Gz

    # checkpoint writer
    checkpoint_dir = logs_dir + '/checkpoints'
    checkpoint = tf.train.Checkpoint(enc_x=enc_x,
                                     dec_z=dec_z)

    # restore from checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    labels = []
    predictions = []
    for img_batch, label_batch in tqdm(data_generator):
        # encoder
        z = enc_x(img_batch, training=False)

        # decoder
        x_hat = dec_z(z, training=False)

        # subtract reconstructed images from originals
        diff_maps = tf.abs(img_batch - x_hat) - 1  # because [-1, 1]

        # transform back to original values and format
        diff_maps = tf.cast((diff_maps + 1) * 127.5, tf.uint8)
        img_batch = tf.cast((img_batch + 1) * 127.5, tf.uint8)

        for orig_img, diff_map, label in zip(img_batch, diff_maps, label_batch):
            pred = detect_anomalies_in_diff_map(diff_map, threshold, min_percentile, min_area,
                                                show_contours, debug, label, orig_img)
            predictions.append(pred)
            labels.append(label)

    return predictions, labels


def detect_anomalies_in_diff_map(diff_map, threshold, min_percentile, min_area, show_contours=True, debug=False,
                                 label=None, orig_img=None):
    # clean diff map from noise
    grey_opening = ndimage.grey_opening(diff_map[:, :, 0], (3, 3), mode='nearest')
    grey_opening = cv2.medianBlur(grey_opening, 3)

    # create masks for min value and min percentile value
    _, mask1 = cv2.threshold(grey_opening, threshold, 255, cv2.THRESH_BINARY)
    img_percentile = np.percentile(grey_opening, min_percentile)
    _, mask2 = cv2.threshold(grey_opening, img_percentile, 255, cv2.THRESH_BINARY)
    masks = mask1 * mask2

    # try to connect close dots
    kernel = np.ones((3, 3))
    thresh_img = cv2.morphologyEx(masks, cv2.MORPH_OPEN, kernel)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

    final_map = np.zeros_like(thresh_img)
    b = 1  # border from edge
    final_map[b:-b, b:-b] = thresh_img[b:-b, b:-b]  # get rid of edge differences (many FP)
    contours, hierarchy = cv2.findContours(final_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_area = [cv2.contourArea(contour) for contour in contours]
    # filter by contour area
    contours = [contour for contour, area in zip(contours, contours_area) if area >= min_area]

    if debug and (orig_img is not None) and (label is not None):
        crop_size = orig_img.shape[1]
        print('Contours area: ', contours_area)
        panels = np.zeros((crop_size, crop_size * 5, 1))
        panels[:, :crop_size, :] = orig_img
        panels[:, crop_size:crop_size * 2, :] = diff_map
        panels[:, crop_size * 2:crop_size * 3, :] = np.expand_dims(grey_opening, -1)
        panels[:, crop_size * 3:crop_size * 4, :] = np.expand_dims(masks * 255, -1)
        panels[:, crop_size * 4:crop_size * 5, :] = np.expand_dims(final_map * 255, -1)

        cv2.imshow(' Image  |  Diff Map  |  Grey Open  |  Masks  |  Final Map  |  Label - {}'.format(str(label)),
                   panels.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    found_contours = (len(contours) > 0)
    if show_contours and (orig_img is not None) and found_contours:
        crop_for_display = orig_img.numpy().astype(np.uint8)
        cv2.drawContours(crop_for_display, contours, -1, (255, 0, 0), 3)
        cv2.imshow(f'GT Label - {str(label)}', crop_for_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 1.0 if found_contours else 0.0
