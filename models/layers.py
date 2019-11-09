import tensorflow as tf


def convolution_layer(images, scope):
    """
    The layer for convolution.
    :param images: Tensor, [batch_size, image_w, image_h, channels].
    :param scope: str, variable scope.
    :return: Tensor, [batch_size, image_w, image_h, -1].
    """
    k_size = images.get_shape()[-1].value
    with tf.variable_scope(f'conv{scope}'):
        weights = tf.get_variable(name=f'weights{scope}',
                                  shape=[5, 5, k_size, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable(name=f'biases{scope}',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv_res = tf.nn.relu(pre_activation)
    return conv_res


def convolution_layer2(images, scope):
    """
    The layer for convolution.
    :param images: Tensor, [batch_size, image_w, image_h, channels].
    :param scope: str, variable scope.
    :return: Tensor, [batch_size, image_w, image_h, -1].
    """
    k_size = images.get_shape()[-1].value
    with tf.variable_scope(f'conv{scope}'):
        weights = tf.get_variable(name=f'weights{scope}',
                                  shape=[5, 5, k_size, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable(name=f'biases{scope}',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv_res = tf.nn.relu(pre_activation)
    return conv_res


def max_and_lrn_layer(conv, scope):
    """
    The layer for max_pooling and local response normalization.
    :param conv: Tensor, [batch_size, image_w, image_h, channels].
    :param scope: str, variable scope.
    :return: Tensor, [batch_size, image_w/2, image_h/2, channels].
    """
    with tf.variable_scope(f'pooling_lrn_{scope}'):
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=f'pooling{scope}')
        norm = tf.nn.lrn(pool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=f'norm{scope}')
    return norm


def full_layer(norm, batch_size, keep_prob):
    """
    The layer for fully connected.
    :param norm: Tensor, [batch_size, image_w, image_h, channels].
    :param batch_size: int, the batch_size of training.
    :param keep_prob: int , the keep_prob of dropout.
    :return:  Tensor, [-1, 64]
    """
    with tf.variable_scope('full_layer1'):
        reshape = tf.reshape(norm, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable(name='full_weights',
                                  shape=[dim, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable(name='full_biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        full_res1 = tf.nn.relu(tf.matmul(reshape, weights) + biases)

    with tf.variable_scope('dropout_layer'):
        drop_res = tf.nn.dropout(full_res1, keep_prob=keep_prob)

    with tf.variable_scope('full_layer2'):
        weights = tf.get_variable(name='full_weights2',
                                  shape=[256, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable(name='full_biases2',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        full_res = tf.nn.relu(tf.matmul(drop_res, weights) + biases)
    with tf.variable_scope('dropout_layer'):
        drop_res = tf.nn.dropout(full_res, keep_prob=keep_prob)
    return drop_res


def soft_max_layer(full_res, n_class):
    """
    The soft_max for output.
    :param full_res: Tensor, [-1, 64].
    :param n_class: int, the number of classes.
    :return: Tensor, [-1, n_class].
    """
    with tf.variable_scope('soft_linear'):
        weights = tf.get_variable(name='soft_weights',
                                  shape=[64, n_class],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable(name='soft_biases',
                                 shape=[n_class],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        soft_res = tf.add(tf.matmul(full_res, weights), biases)
    return soft_res


def out_put_layer(images, args):
    """
    The combination of all layers.
    :param images: Tensor, [batch_size, image_w, image_h, channels].
    :param args: The instance of argparse, arguments for training.
    :return: Tensor, [-1, n_class].
    """
    conv1 = convolution_layer(images, 1)
    max1 = max_and_lrn_layer(conv1, 1)
    conv2 = convolution_layer2(max1, 2)
    max2 = max_and_lrn_layer(conv2, 2)
    full = full_layer(max2, args.batch_size, args.keep_prob)
    res = soft_max_layer(full, args.n_class)
    return res
