import tensorflow as tf

tf.debugging.set_log_device_placement(True)

if __name__ == '__main__':
    with tf.compat.v1.Session() as sess:
        with tf.device('/gpu:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            c = tf.matmul(a, b)

        print(sess.run(c))
