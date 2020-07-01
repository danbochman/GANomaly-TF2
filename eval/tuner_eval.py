import os

import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from eval.eval_utils import save_precision_recall_curve
from tqdm import tqdm
from dataloader.image_generators import test_image_generator
from models.autoencoders import CAE
from models.tuners import FilterSearcher

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def eval_tuner(tuner, test_img_gen):
    labels = []
    anomaly_scores = []
    for cogwheel_crop, label in tqdm(test_img_gen):
        labels.extend(label)
        anomaly_score = tuner(cogwheel_crop)
        anomaly_scores.extend(anomaly_score)

    return anomaly_scores, labels

def main():
    crop_size = 128
    latent_dim = 64
    batch_size = 128

    defect_data_path = "/media/jpowell/hdd/Data/AIS/RO2_NG_images/"

    test_img_gen = test_image_generator(defect_data_path, batch_size=batch_size, crop_size=crop_size, preprocess=False,
                                        repeat=False)

    print('Initializing autoencoder model...')
    input_shape = (crop_size, crop_size, 1)
    cae = CAE(input_shape=input_shape, latent_dim=latent_dim)
    path_to_ae_model = '/home/jpowell/PycharmProjects/AIS/ais_aae/train/RO2_AC_128x_64d_best_model.h5'
    if os.path.exists(path_to_ae_model):
        print('Loading model from checkpoint....')
        cae.load_weights(path_to_ae_model)
        for layer in cae.layers:
            layer.trainable = False
        cae.compile()
        print('autoencoder loaded, frozen and compiled')

    print('Initializing tuned model...')
    tuner_model = FilterSearcher(input_shape, cae)
    path_to_tuner_model = '/home/jpowell/PycharmProjects/AIS/ais_aae/train/RO2_Tuner.h5'
    if os.path.exists(path_to_ae_model):
        print('Loading tuner from checkpoint....')
        tuner_model.load_weights(path_to_tuner_model)
        for layer in cae.layers:
            layer.trainable = False
        tuner_model.compile()
        print('tuner loaded, frozen and compiled')

    anomaly_scores, labels = eval_tuner(tuner_model, test_img_gen)
    save_precision_recall_curve(anomaly_scores, labels)




if __name__ == '__main__':
    main()
