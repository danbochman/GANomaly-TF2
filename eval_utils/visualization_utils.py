import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import compare_ssim
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay


def save_precision_recall_curve(anomaly_scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, anomaly_scores)
    average_precision = average_precision_score(labels, anomaly_scores)
    PrecisionRecallDisplay(precision, recall, average_precision, 'CAE').plot()
    plt.show()
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


def show_ssim(inputs, reconstructed, crop_size, labels=None):
    for i in range(inputs.shape[0]):
        image = inputs[i]
        reconst = reconstructed[i].numpy().astype(np.float64)
        score, diff = compare_ssim(image[:, :, 0], reconst[:, :, 0], gaussian_weights=True, full=True)
        diff = np.expand_dims((diff * crop_size).astype("uint8"), axis=-1)
        triptych = np.zeros((256, crop_size * 3, 1))
        triptych[:, :crop_size, :] = inputs[i]
        triptych[:, crop_size:crop_size * 2, :] = reconstructed[i]
        triptych[:, crop_size * 2:crop_size * 3, :] = diff

        label = 'Unknown'
        if labels is not None:
            label = str(labels[i])
        cv2.imshow(f'Original     |     Reconstructed     |     SSIM Difference Map     |     Label - {label}',
                   triptych.astype(np.uint8))

        cv2.waitKey(0)
