import cv2
import numpy as np
from skimage.measure import compare_ssim


def show_heatmap(inputs, diff_map, labels):
    for i in range(inputs.shape[0]):
        comparison = np.zeros((256, 512, 3))
        comparison[:, :256, :] = inputs[i]
        overlay = (inputs[i] * 0.7) + (diff_map[i] * 0.3)
        comparison[:, 256:512:, :] = cv2.applyColorMap(overlay.numpy().astype(np.uint8), cv2.COLORMAP_JET)
        label = 'Unknown'
        if labels is not None:
            label = str(labels[i])
        cv2.imshow(f'Original   |     Difference Heatmap     |     Label - {label}',
                   comparison.astype(np.uint8))
        cv2.waitKey(0)


def show_triptych(inputs, reconstructed, diff_map, labels=None):
    for i in range(inputs.shape[0]):
        triptych = np.zeros((256, 256 * 3, 1))
        triptych[:, :256, :] = inputs[i]
        triptych[:, 256:512, :] = reconstructed[i]
        triptych[:, 512:768, :] = diff_map[i]
        label = 'Unknown'
        if labels is not None:
            label = str(labels[i])
        cv2.imshow(f'Original   |     Reconstructed    |     Difference      |     Label - {label}',
                   triptych.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_ssim(inputs, reconstructed, labels=None):
    for i in range(inputs.shape[0]):
        image = inputs[i]
        reconst = reconstructed[i].numpy().astype(np.float64)
        score, diff = compare_ssim(image[:, :, 0], reconst[:, :, 0], gaussian_weights=True, full=True)
        diff = np.expand_dims((diff * 255).astype("uint8"), axis=-1)
        triptych = np.zeros((256, 256 * 3, 1))
        triptych[:, :256, :] = inputs[i]
        triptych[:, 256:512, :] = reconstructed[i]
        triptych[:, 512:768, :] = diff

        label = 'Unknown'
        if labels is not None:
            label = str(labels[i])
        cv2.imshow(f'Original     |     Reconstructed     |     SSIM Difference Map     |     Label - {label}',
                   triptych.astype(np.uint8))

        cv2.waitKey(0)