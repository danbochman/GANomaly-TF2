import numpy as np


def center_and_scale(img):
    img = img / 255
    img = 2 * img - 1
    return img


def stack_and_expand(img_lst):
    img_slices = np.stack(img_lst, axis=0)
    img_slices = np.expand_dims(img_slices, axis=-1).astype(np.float)
    return img_slices
