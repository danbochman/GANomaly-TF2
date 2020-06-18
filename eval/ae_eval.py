import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from tqdm import tqdm

from dataloader.image_generators import test_image_generator
from models.autoencoders import CAE


def eval_cae_anomaly_scores(path_to_data, path_to_weights, first_run=True):
    if first_run:
        crop_size = 256
        test_img_gen = test_image_generator(path_to_data, crop_size=crop_size)

        cae = CAE(input_shape=(crop_size, crop_size, 1))

        if os.path.exists(path_to_weights):
            cae.load_weights(path_to_weights)
        anomaly_scores_total = []
        labels_total = []
        for cogwheel_crops, labels in tqdm(test_img_gen):
            anomaly_scores = cae.anomaly_scores(cogwheel_crops)
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


def eval_cae_detect_anomalies(path_to_data, path_to_weights, first_run=True):
    if first_run:
        crop_size = 256
        test_img_gen = test_image_generator(path_to_data, batch_size=1, crop_size=crop_size)

        cae = CAE(input_shape=(crop_size, crop_size, 1))

        if os.path.exists(path_to_weights):
            cae.load_weights(path_to_weights)

        for cogwheel_crop, label in tqdm(test_img_gen):
            crop_for_display = cogwheel_crop[0].astype(np.uint8)
            if label == 1:
                cv2.imshow('Original', crop_for_display)
                cv2.waitKey(0)
                bboxes = cae.detect_anomalies(cogwheel_crop)
                if len(bboxes) > 0:
                    cv2.drawContours(crop_for_display, bboxes, -1, (255, 0, 0), 3)
                    cv2.imshow(f'GT Label - {str(label)}', crop_for_display)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


def save_precision_recall_curve(anomaly_scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, anomaly_scores)
    average_precision = average_precision_score(labels, anomaly_scores)
    PrecisionRecallDisplay(precision, recall, average_precision, 'CAE').plot()
    plt.savefig('precision_recall_curve.png', dpi=400)


def main():
    data_path = "D:/Razor Labs/Projects/AIS/data/RO2/RO2_NG_images/by_cams/"
    path_to_weights = 'D:\\Users\\danbo\\PycharmProjects\\AIS\\train\\best_weights.h5'
    anomaly_scores, labels = eval_cae_anomaly_scores(data_path, path_to_weights, first_run=True)
    # eval_cae_detect_anomalies(data_path, path_to_weights, first_run=True)
    save_precision_recall_curve(anomaly_scores, labels)


if __name__ == '__main__':
    main()
