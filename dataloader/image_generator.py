import cv2
import numpy as np

from dataloader.cogwheel_slicer import img_slicer, img_roi

RO2_BOUNDS = (20, 200)  # upper bound, lower bound
RO2_PERIOD = 200


def image_generator(crop_size=256):
    img_path = "D:/Razor Labs/Projects/AIS/data/RO2/RO2_OK_images/Cam1/img/PART1_PART1_Cam1_IO__23440-R02-C000_right_000154.png"
    img = cv2.imread(img_path, 0)
    img = img_roi(img, *RO2_BOUNDS)
    img_slices = img_slicer(img, crop_size)
    img_slices = np.stack(img_slices[:-1], axis=0)
    img_slices = np.expand_dims(img_slices, axis=-1).astype(np.float)
    while True:
        yield img_slices, img_slices
