import glob
from random import shuffle, seed, sample

import cv2
import numpy as np

from dataloader.annotation_utils import load_annotation_file, annotation_to_bboxes_ltwh
from dataloader.cogwheel_slicer import img_slice_and_label
from dataloader.preprocessing import stack_and_expand


def crop_generator(img_list, batch_size=64, crop_size=256, preprocess=False, repeat=True):
    repeats = 2 ** 32 if repeat else 1
    for i in range(repeats):
        for img_path in img_list:
            img = cv2.imread(img_path, 0)
            img_slices, _ = img_slice_and_label(img, crop_size, preprocess=preprocess)  # _ because no labels
            img_slices = stack_and_expand(img_slices)
            for i in range(0, img_slices.shape[0], batch_size):
                img_batch = img_slices[i:i + batch_size]
                yield img_batch, img_batch


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


def train_val_image_generator(folder_path, batch_size=128, crop_size=128, ext="png",
                              val_ratio=0.2, preprocess=False, repeat=True, random_state=0, sample_frac=None):
    seed(random_state)
    img_list = [img for img in sorted(glob.glob(folder_path + "**/*." + ext, recursive=True))]
    ann_list = [img for img in sorted(glob.glob(folder_path + "**/*." + "json", recursive=True))]
    normal_img_list, defect_img_list = separate_images_by_label(img_list, ann_list)
    normal_imgs, _ = normal_img_list  # no need for annotations
    shuffle(normal_imgs)
    num_images = len(normal_imgs)
    if sample_frac is not None:
        normal_imgs = sample(normal_imgs, int(num_images * sample_frac))
    train_val_split = int(num_images * (1 - val_ratio))
    train_imgs = normal_imgs[:train_val_split]
    val_imgs = normal_imgs[train_val_split:]
    train_generator = crop_generator(train_imgs, batch_size=batch_size, crop_size=crop_size, preprocess=preprocess,
                                     repeat=repeat)
    val_generator = crop_generator(val_imgs, batch_size=batch_size, crop_size=crop_size, preprocess=preprocess,
                                   repeat=repeat)
    return train_generator, val_generator


def test_image_generator(folder_path, batch_size=128, crop_size=128, ext="png", preprocess=False):
    img_list = [img for img in sorted(glob.glob(folder_path + "**/*." + ext, recursive=True))]
    ann_list = [img for img in sorted(glob.glob(folder_path + "**/*." + "json", recursive=True))]
    normal_img_list, defect_img_list = separate_images_by_label(img_list, ann_list)
    # defect_imgs, defect_ann = list(zip(*defect_img_list))

    # ann_list.pop(0)  # drop meta.json
    for img_path, ann_path in defect_img_list:
        img = cv2.imread(img_path, 0)
        ann = load_annotation_file(ann_path)
        bboxes = annotation_to_bboxes_ltwh(ann)
        img_slices, labels = img_slice_and_label(img, crop_size, bboxes=bboxes, preprocess=preprocess)
        img_slices = stack_and_expand(img_slices)
        labels = np.array(labels)
        for i in range(0, img_slices.shape[0], batch_size):
            img_batch = img_slices[i:i + batch_size]
            label_batch = labels[i:i + batch_size]
            yield img_batch, label_batch


if __name__ == '__main__':
    train_img_gen, val_img_gen = train_val_image_generator("/media/jpowell/hdd/Data/AIS/RO2_OK_images/")
    for i in range(10):
        train_img_sample = train_img_gen.__next__()
        val_img_sample = val_img_gen.__next__()
        print(train_img_sample[0].shape)
        print(val_img_sample[0].shape)
        print(' ')
