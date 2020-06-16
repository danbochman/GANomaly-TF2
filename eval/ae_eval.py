import os
from dataloader.image_generators import test_image_generator
from models.autoencoders import CAE
from train.losses import reconstruction_mse


def main():
    crop_size = 256
    test_img_gen = test_image_generator("D:/Razor Labs/Projects/AIS/data/RO2/RO2_NG_images/", crop_size=crop_size)

    cae = CAE(input_shape=(crop_size, crop_size, 1))

    path_to_weights = 'D:\\Users\\danbo\\PycharmProjects\\AIS\\train\\best_weights.h5'

    if os.path.exists(path_to_weights):
        cae.load_weights(path_to_weights)

    for i in range(10):
        samples, labels = test_img_gen.__next__()
        cae.visualize_anomalies(samples, labels=labels)



if __name__ == '__main__':
    main()
