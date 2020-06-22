import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from tqdm import tqdm

from dataloader.image_generators import test_image_generator, train_image_generator
from models.autoencoders import CAE
from train.losses import mse_ssim_mixed

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def eval_cae_anomaly_scores(path_to_data, path_to_weights, batch_size=64, first_run=True):
    if first_run:
        crop_size = 256
        test_img_gen = test_image_generator(path_to_data, batch_size=batch_size, crop_size=crop_size)

        cae = CAE(input_shape=(crop_size, crop_size, 1))

        if os.path.exists(path_to_weights):
            cae.load_weights(path_to_weights)

        anomaly_scores_total = []
        labels_total = []
        for cogwheel_crops, labels in tqdm(test_img_gen):
            cogwheel_crops = tf.convert_to_tensor(cogwheel_crops, dtype=tf.float32)
            anomaly_scores = cae.anomaly_scores(cogwheel_crops, metric_fn=mse_ssim_mixed)
            anomaly_scores_total.extend(anomaly_scores)
            labels_total.extend(labels)

        anomaly_scores_total = np.stack(anomaly_scores_total, axis=0)
        labels_total = np.stack(labels_total, axis=0)

        np.savez('eval_results.npz', anomaly_scores=anomaly_scores_total, labels=labels_total)

    else:
        saved_data = np.load('eval_results.npz')
        anomaly_scores_total = saved_data['anomaly_scores']
        labels_total = saved_data['labels']

    return anomaly_scores_total, labels_total


def eval_cae_detect_anomalies_by_crop(path_to_data, path_to_weights, display=False, first_run=True):
    if first_run:
        crop_size = 256
        test_img_gen = test_image_generator(path_to_data, batch_size=1, crop_size=crop_size)

        cae = CAE(input_shape=(crop_size, crop_size, 1))

        if os.path.exists(path_to_weights):
            cae.load_weights(path_to_weights)

        labels = []
        predictions = []
        for cogwheel_crop, label in tqdm(test_img_gen):
            labels.extend(label)
            crop_for_display = cogwheel_crop[0].astype(np.uint8)
            bboxes = cae.detect_anomalies(cogwheel_crop, display=display)
            pred = (len(bboxes) > 0)
            predictions.append(pred)
            if pred:
                if display:
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


def eval_cae_detect_anomalies_by_images(path_to_normal_data, path_to_defect_data, path_to_weights,
                                        display=False,
                                        first_run=True):
    if first_run:
        crop_size = 256
        defect_img_gen = test_image_generator(path_to_defect_data, batch_size=64, crop_size=crop_size)
        normal_img_gen = train_image_generator(path_to_normal_data, batch_size=64, crop_size=crop_size, repeat=False)

        cae = CAE(input_shape=(crop_size, crop_size, 1))

        if os.path.exists(path_to_weights):
            cae.load_weights(path_to_weights)

        agg_labels = []
        agg_predictions = []
        for cogwheel_crops, _ in tqdm(defect_img_gen):
            agg_labels.append(1)
            pred = 0
            for crop in cogwheel_crops:
                crop = np.expand_dims(crop, axis=0)
                bboxes = cae.detect_anomalies(crop, display=display)
                if len(bboxes) > 0:
                    pred = 1
                    break
            agg_predictions.append(pred)

        for cogwheel_crops, _ in tqdm(normal_img_gen):
            agg_labels.append(0)
            pred = 0
            for crop in cogwheel_crops:
                crop = np.expand_dims(crop, axis=0)
                bboxes = cae.detect_anomalies(crop, display=display)
                if len(bboxes) > 0:
                    pred = 1
                    break
            agg_predictions.append(pred)

        predictions = np.stack(agg_predictions, axis=0)
        labels = np.stack(agg_labels, axis=0)
        np.savez('det_by_image_eval_results.npz', predictions=predictions, labels=labels)

    else:
        saved_data = np.load('det_by_image_eval_results.npz')
        predictions = saved_data['predictions']
        labels = saved_data['labels']

    return predictions, labels


def save_precision_recall_curve(anomaly_scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, anomaly_scores)
    average_precision = average_precision_score(labels, anomaly_scores)
    PrecisionRecallDisplay(precision, recall, average_precision, 'CAE').plot()
    plt.show()
    plt.savefig('precision_recall_curve.png', dpi=400)


def visualize_diff(path_to_images, path_to_weights, method='heatmap'):
    crop_size = 256
    test_img_gen = test_image_generator(path_to_images, batch_size=64, crop_size=crop_size)

    cae = CAE(input_shape=(crop_size, crop_size, 1))

    if os.path.exists(path_to_weights):
        cae.load_weights(path_to_weights)

    for cogwheel_crops, labels in test_img_gen:
        cae.visualize_anomalies(cogwheel_crops, method=method, labels=labels)


def main():
    defect_data_path = "/media/jpowell/hdd/Data/AIS/RO2_NG_images/"
    normal_data_path = "/media/jpowell/hdd/Data/AIS/RO2_OK_images/"
    path_to_weights = '/home/jpowell/PycharmProjects/AIS/ais_aae/train/ssim_mse_mixed_best_weights.h5'

    # visualize_diff(data_path, path_to_weights, method='heatmap')

    # anomaly_scores, labels = eval_cae_anomaly_scores(defect_data_path, path_to_weights, first_run=True)
    # save_precision_recall_curve(anomaly_scores, labels)

    predictions, labels = eval_cae_detect_anomalies_by_images(normal_data_path, defect_data_path, path_to_weights,
                                                              first_run=True)
    cm = confusion_matrix(labels, predictions)

    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly']).plot()
    plt.show()


if __name__ == '__main__':
    main()
