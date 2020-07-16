import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay


def display_precision_recall_curve(anomaly_scores, labels, save_png=False):
    precision, recall, thresholds = precision_recall_curve(labels, anomaly_scores)
    average_precision = average_precision_score(labels, anomaly_scores)
    PrecisionRecallDisplay(precision, recall, average_precision, 'CAE').plot()
    plt.show()
    if save_png:
        plt.savefig('precision_recall_curve.png', dpi=400)
    return precision, recall, thresholds


def show_heatmap(inputs, diff_map, crop_size, labels=None):
    for i in range(inputs.shape[0]):
        comparison = np.zeros((crop_size, crop_size, 3))
        comparison[:, :crop_size, :] = inputs[i]
        overlay = (inputs[i] * 0.7) + (diff_map[i] * 0.3)
        comparison[:, crop_size:crop_size * 2:, :] = cv2.applyColorMap(overlay.numpy().astype(np.uint8),
                                                                       cv2.COLORMAP_JET)
        label = 'Unknown'
        if labels is not None:
            label = str(labels[i])
        cv2.imshow(f'Original   |     Difference Heatmap     |     Label - {label}',
                   comparison.astype(np.uint8))
        cv2.waitKey(0)


def show_triptych(inputs, reconstructed, diff_map, crop_size, labels=None):
    for i in range(inputs.shape[0]):
        triptych = np.zeros((crop_size, int(crop_size * 3), 1))
        triptych[:, :crop_size, :] = inputs[i]
        triptych[:, crop_size:crop_size * 2, :] = reconstructed[i]
        triptych[:, crop_size * 2:crop_size * 3, :] = diff_map[i]
        label = 'Unknown'
        if labels is not None:
            label = str(labels[i])
        cv2.imshow(f'Original   |     Reconstructed    |     Difference      |     Label - {label}',
                   triptych.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_histogram_and_pr_curve(anomaly_scores, labels):
    # normalize anomaly scores
    anomaly_scores = np.array(anomaly_scores)
    as_max = np.max(anomaly_scores)
    as_min = np.min(anomaly_scores)
    anomaly_scores = (anomaly_scores - as_min) / (as_max - as_min)

    # plot histogram w.r.t label
    labels = np.array(labels)
    normal = anomaly_scores[labels == 0]
    defect = anomaly_scores[labels == 1]
    plt.hist(normal, bins=100, alpha=0.5, label='normal', density=True)
    plt.hist(defect, bins=100, alpha=0.5, label='defect', density=True)
    plt.legend(loc='upper right')
    plt.show()

    # plot precision recall curve
    display_precision_recall_curve(anomaly_scores, labels)


def display_confusion_matrix(predictions, labels):
    cm = confusion_matrix(labels, predictions)
    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly']).plot(values_format='d')
    plt.show()
