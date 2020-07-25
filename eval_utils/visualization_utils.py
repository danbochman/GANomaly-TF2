import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay


def show_histogram_and_pr_curve(anomaly_scores, labels):
    """
    This function will help you visualize how good your anomaly scores separate your normal and anomalous images by
    plotting the distributions of the scores w.r.t to the labels (anomaly = 1, normal = 0). In addition, you will
    see the sensitivity to the score threshold for predictions in the precision-recall curve.
    """
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


def display_precision_recall_curve(anomaly_scores, labels, save_png=False):
    precision, recall, thresholds = precision_recall_curve(labels, anomaly_scores)
    average_precision = average_precision_score(labels, anomaly_scores)
    PrecisionRecallDisplay(precision, recall, average_precision, 'CAE').plot()
    plt.show()
    if save_png:
        plt.savefig('precision_recall_curve.png', dpi=400)
    return precision, recall, thresholds


def display_confusion_matrix(predictions, labels):
    cm = confusion_matrix(labels, predictions)
    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly']).plot(values_format='d')
    plt.show()
