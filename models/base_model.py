import tensorflow as tf
from models.layers import *
from os.path import join


def build_model(images, args):
    """
    The forward propagation of model.
    :param images: Tensor, [batch_size, image_w, image_h, channels]
    :param args: The instance of argparse, arguments for training.
    :return: Tensor, [-1, n_class]
    """
    logits_y = out_put_layer(images, args)
    return logits_y


def losses(logits_y, labels):
    """
    Compute the loss of this epoch.
    :param logits_y: Tensor, [-1, n_class]
    :param labels: Tensor, [n_class]
    :return: Tensor
    """
    reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_y, labels=labels,
                                                                   name='entropy_per_example')
    train_loss = tf.reduce_mean(cross_entropy, name='loss') + reg
    return train_loss


def prediction(logits_y):
    """
    Predict the label of each sample.
    :param logits_y: Tensor, [-1, n_class]
    :return: Tensor
    """
    pre_y = tf.argmax(logits_y, 1, name='pre_Y')
    return pre_y


def train(loss, learning_rate, global_step):
    """
    The back propagation of the model.
    :param loss: Tensor
    :param learning_rate: float, the learning_rate of training.
    :param global_step: int
    :return: An Operation that updates the variables in `var_list`.
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy'):
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    return accuracy, labels

