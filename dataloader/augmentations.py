import tensorflow as tf

AUGMENTATIONS = [
    lambda x: tf.image.random_flip_left_right(x),
    lambda x: tf.image.random_brightness(x, 0.3),
    lambda x: tf.image.random_contrast(x, 0, 3),
]


def augment(images, prob_to_augment=0.25):
    for augmentation_fn in AUGMENTATIONS:
        prob = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
        if prob > prob_to_augment:
            images = augmentation_fn(images)

    return images
