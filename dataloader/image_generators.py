import glob
import random

import cv2
import numpy as np

from dataloader.annotation_utils import load_annotation_file, annotation_to_bboxes_ltwh
from dataloader.cogwheel_slicer import img_slice_and_label
from dataloader.preprocessing import stack_and_expand, center_and_scale


def crop_generator(img_ann_list, batch_size=64, crop_size=256, shuffle=True, normalize=True, repeat=True):
    repeats = 2 ** 32 if repeat else 1
    random.seed(42)

    for i in range(repeats):

        if shuffle:
            random.shuffle(list(img_ann_list))

        for img_path, ann_path in img_ann_list:
            img = cv2.imread(img_path, 0)
            if normalize:
                img = center_and_scale(img)
            ann = load_annotation_file(ann_path)
            bboxes = annotation_to_bboxes_ltwh(ann)
            img_slices, labels = img_slice_and_label(img, crop_size, bboxes=bboxes)
            img_slices = stack_and_expand(img_slices)
            labels = np.array(labels)
            for i in range(0, img_slices.shape[0], batch_size):
                img_batch = img_slices[i:i + batch_size]
                label_batch = labels[i:i + batch_size]
                yield img_batch, label_batch


def separate_images_by_label(img_list, ann_list):
    normal_img_list = []
    defect_img_list = []
    for img_path, ann_path, in zip(img_list, ann_list):
        ann_parsed = load_annotation_file(ann_path)
        bboxes = annotation_to_bboxes_ltwh(ann_parsed)
        if len(bboxes) == 0:
            normal_img_list.append((img_path, ann_path))
        elif len(bboxes) > 0:
            defect_img_list.append((img_path, ann_path))

    return normal_img_list, defect_img_list


def train_val_test_image_generator(folder_path, batch_size=128, crop_size=128, ext="png", normalize=True, val_frac=0.0):
    # load img and annotation filepath recursively from folder
    img_list = [img for img in sorted(glob.glob(folder_path + "**/*." + ext, recursive=True))]
    ann_list = [img for img in sorted(glob.glob(folder_path + "**/*." + "json", recursive=True))]

    # separate images/annotation by label
    normal_img_ann_list, defect_img_ann_list = separate_images_by_label(img_list, ann_list)

    # split to train/val/test
    test_generator = crop_generator(defect_img_ann_list, batch_size=batch_size, crop_size=crop_size,
                                    normalize=normalize,
                                    repeat=False,
                                    shuffle=False)
    if val_frac:
        # split to train/val (we only need normal imgs)
        num_images = len(normal_img_ann_list)
        train_val_split = int(num_images * (1 - val_frac))
        train_img_ann_list = normal_img_ann_list[:train_val_split]
        val_img_ann_list = normal_img_ann_list[train_val_split:]

        train_generator = crop_generator(train_img_ann_list, batch_size=batch_size, crop_size=crop_size,
                                         normalize=normalize,
                                         repeat=True,
                                         shuffle=True)
        val_generator = crop_generator(val_img_ann_list, batch_size=batch_size, crop_size=crop_size,
                                       normalize=normalize,
                                       repeat=True,
                                       shuffle=False)

        return train_generator, val_generator, test_generator

    else:
        train_generator = crop_generator(normal_img_ann_list, batch_size=batch_size, crop_size=crop_size,
                                         normalize=normalize,
                                         repeat=True,
                                         shuffle=True)

        return train_generator, test_generator


if __name__ == '__main__':
    data_path = "/media/jpowell/hdd/Data/AIS/RO2_NG_images/"
    train_generator, val_generator, test_generator = train_val_test_image_generator(data_path, val_frac=0.2)
    for i in range(10):
        train_img_sample = train_generator.__next__()
        val_img_sample = val_generator.__next__()
        test_img_sample = test_generator.__next__()
        print(train_img_sample[0].shape)
        print(val_img_sample[0].shape)
        print(test_img_sample[0].shape)
