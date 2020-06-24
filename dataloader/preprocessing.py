import cv2
import numpy as np


def equalize_and_smooth(img, clipLimit=3.0, eq_win=(8, 8), median_size=3):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=eq_win)
    crop_clahe = clahe.apply(img)
    smoothed = cv2.medianBlur(crop_clahe, median_size)
    return smoothed


def stack_and_expand(img_lst):
    img_slices = np.stack(img_lst, axis=0)
    img_slices = np.expand_dims(img_slices, axis=-1).astype(np.float)
    return img_slices
