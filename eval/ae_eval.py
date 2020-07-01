import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from tqdm import tqdm

from dataloader.image_generators import test_image_generator, train_val_image_generator
from models.autoencoders import CAE

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def eval_cae_anomaly_scores(path_to_data, path_to_weights, metric_fn, batch_size=1, crop_size=128, latent_dim=256,
                            first_run=True):
    if first_run:
        test_img_gen = test_image_generator(path_to_data, batch_size=batch_size, crop_size=crop_size)

        cae = CAE(input_shape=(crop_size, crop_size, 1), latent_dim=latent_dim)

        if os.path.exists(path_to_weights):
            cae.load_weights(path_to_weights)

        anomaly_scores_total = []
        labels_total = []
        for cogwheel_crops, labels in tqdm(test_img_gen):
            cogwheel_crops = tf.convert_to_tensor(cogwheel_crops, dtype=tf.float32)
            anomaly_scores = cae.anomaly_scores(cogwheel_crops, metric_fn=metric_fn)
            anomaly_scores_total.extend(anomaly_scores)
            labels_total.extend(labels)

    return anomaly_scores_total, labels_total


def eval_cae_detect_anomalies_by_crop(path_to_defects_data, path_to_weights, min_area=9,
                                      latent_dim=256, crop_size=128, batch_size=1,
                                      preprocess=False, debug=False, first_run=True):
    if first_run:
        test_img_gen = test_image_generator(path_to_defects_data, batch_size=1, crop_size=crop_size,
                                            preprocess=preprocess, )

        cae = CAE(input_shape=(crop_size, crop_size, 1), latent_dim=latent_dim)

        if os.path.exists(path_to_weights):
            cae.load_weights(path_to_weights)

        labels = []
        predictions = []
        for cogwheel_crop, label in tqdm(test_img_gen):
            labels.extend(label)
            crop_for_display = cogwheel_crop[0].astype(np.uint8)
            bboxes = cae.detect_anomalies(cogwheel_crop, label=label, debug=debug, crop_size=crop_size,
                                          min_area=min_area)
            pred = (len(bboxes) > 0)
            predictions.append(pred)
            if pred:
                if debug and label == 1:
                    cv2.drawContours(crop_for_display, bboxes, -1, (255, 0, 0), 3)
                    cv2.imshow(f'GT Label - {str(label)}', crop_for_display)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        predictions = np.stack(predictions, axis=0)
        labels = np.stack(labels, axis=0)
        np.savez('det_by_crop_eval_results.npz', predictions=predictions, labels=labels)

    else:
        saved_data = np.load('det_eval_results.npz')
        predictions = saved_data['predictions']
        labels = saved_data['labels']

    return predictions, labels





def visualize_diff(path_to_images, path_to_weights, method='heatmap', latent_dim=256, crop_size=128, batch_size=128):
    test_img_gen = test_image_generator(path_to_images, batch_size=128, crop_size=crop_size, preprocess=False)

    cae = CAE(input_shape=(crop_size, crop_size, 1), latent_dim=latent_dim)

    if os.path.exists(path_to_weights):
        cae.load_weights(path_to_weights)

    for cogwheel_crops, labels in test_img_gen:
        cae.visualize_anomalies(cogwheel_crops, method=method, labels=labels, crop_size=crop_size)


def find_median_threshold(path_to_images, path_to_weights, metric_fn, crop_size=128, latent_dim=256,
                          sample_frac=None,
                          batch_size=1):
    img_gen, _ = train_val_image_generator(path_to_images, batch_size=1, val_ratio=0.0, crop_size=crop_size,
                                           repeat=False, sample_frac=sample_frac)

    cae = CAE(input_shape=(crop_size, crop_size, 1), latent_dim=latent_dim)

    if os.path.exists(path_to_weights):
        cae.load_weights(path_to_weights)

    losses = []
    for cogwheel_crops, _ in tqdm(img_gen):
        cogwheel_crops = tf.convert_to_tensor(cogwheel_crops, dtype=tf.float32)
        reconstructed = cae(cogwheel_crops)
        loss = metric_fn(cogwheel_crops, reconstructed)
        losses.append(loss)

    losses = np.array(losses)
    plt.hist(losses, bins=50, range=(0, 500), density=True)
    plt.show()

    return np.median(losses)


def main():
    # defect_data_path = "/media/jpowell/hdd/Data/AIS/8C3W_per_Camera/"
    defect_data_path = "/media/jpowell/hdd/Data/AIS/RO2_NG_images/"
    # normal_data_path = "/media/jpowell/hdd/Data/AIS/RO2_OK_images/"
    # path_to_weights = '/home/jpowell/PycharmProjects/AIS/ais_aae/train/8C3W_128x_256d_best_model.h5'
    path_to_weights = '/home/jpowell/PycharmProjects/AIS/ais_aae/train/RO2_AC_128x_64d_best_model.h5'

    nn_params = {
        'crop_size': 128,  # 128 for RO2, 256 for 8C3W
        'latent_dim': 64,
        'batch_size': 128
    }

    # median_threshold = find_median_threshold(normal_data_path, path_to_weights, mse_ssim_mixed,
    #                                          sample_frac=0.01, **nn_params)
    # print(median_threshold)

    # visualize_diff(defect_data_path, path_to_weights, method='triptych', **nn_params)

    # anomaly_scores, labels = eval_cae_anomaly_scores(defect_data_path, path_to_weights,
    #                                                  metric_fn=mse_dssim_mixed,
    #                                                  first_run=True,
    #                                                  **nn_params)
    # save_precision_recall_curve(anomaly_scores, labels)

    predictions, labels = eval_cae_detect_anomalies_by_crop(defect_data_path,
                                                            path_to_weights,
                                                            preprocess=False,
                                                            first_run=True,
                                                            debug=False,
                                                            min_area=10,
                                                            **nn_params)

    cm = confusion_matrix(labels, predictions)
    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly']).plot()
    plt.show()


if __name__ == '__main__':
    main()
