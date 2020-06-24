import cv2
import numpy as np

from dataloader.image_generators import test_image_generator

if __name__ == '__main__':
    defect_data_path = "/media/jpowell/hdd/Data/AIS/RO2_NG_images/"

    crop_size = 256
    test_img_gen = test_image_generator(defect_data_path, batch_size=1, crop_size=crop_size)

    for cogwheel_crop, label in test_img_gen:
        if label == 1:
            original_crop = cogwheel_crop[0].astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            crop_clahe = clahe.apply(original_crop)
            smoothed = cv2.medianBlur(crop_clahe, 3)
            cv2.imshow('Original', original_crop)
            cv2.imshow('Equalized', crop_clahe)
            cv2.imshow('Smoothed', smoothed)
            cv2.waitKey(0)
