import glob
from random import shuffle

import cv2
import numpy as np

from dataloader.cogwheel_slicer import img_slicer, img_slice_and_label
from dataloader.annotation_utils import load_annotation_file, annotation_to_bboxes_ltwh

RO2_BOUNDS = (20, 200)  # upper bound, lower bound
RO2_PERIOD = 200


def train_image_generator(folder_path, crop_size=256, ext="png"):
    img_list = [img for img in glob.glob(folder_path + "**/*." + ext, recursive=True)]
    shuffle(img_list)
    while True:
        for img_path in img_list:
            img = cv2.imread(img_path, 0)
            img_slices = img_slicer(img, crop_size)
            img_slices = np.stack(img_slices, axis=0)
            img_slices = np.expand_dims(img_slices, axis=-1).astype(np.float)
            yield img_slices, img_slices


def test_image_generator(folder_path, crop_size=256, ext="png"):
    img_list = [img for img in glob.glob(folder_path + "**/*." + ext, recursive=True)]
    ann_list = [img for img in glob.glob(folder_path + "**/*." + "json", recursive=True)]
    ann_list.pop(0)  # drop meta.json

    while True:
        for img_path, ann_path in zip(img_list, ann_list):
            img = cv2.imread(img_path, 0)
            ann = load_annotation_file(ann_path)
            bboxes = annotation_to_bboxes_ltwh(ann)
            img_slices, labels = img_slice_and_label(img, crop_size, bboxes)
            img_slices = np.stack(img_slices, axis=0)
            img_slices = np.expand_dims(img_slices, axis=-1).astype(np.float)
            labels = np.array(labels)
            yield img_slices, labels


if __name__ == '__main__':
    # train_img_gen = train_image_generator("D:/Razor Labs/Projects/AIS/data/RO2/RO2_OK_images/")
    # for i in range(10):
    #     img_sample = train_img_gen.__next__()
    test_img_gen = test_image_generator("D:/Razor Labs/Projects/AIS/data/RO2/RO2_NG_images/")
    for i in range(10):
        img_samples, labels = test_img_gen.__next__()
        defects = img_samples[labels == 1]
        pass


