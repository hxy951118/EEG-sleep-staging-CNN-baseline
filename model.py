import tensorflow as tf

class model():

    def inference(self,images, batch_size, n_classes, dropout_keep_prob):
        '''Build the model
        Args:
            images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
        Returns:
            output tensor with the computed logits, float, [batch_size, n_classes]
        '''
        # conv1, shape = [kernel size, kernel size, channels, kernel numbers]
        # l2_loss = tf.constant(0.0)
        with tf.device('/gpu:0'),tf.variable_scope('conv1') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3, 3, 1, 32],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[32],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

        # pool1 and norm1
        with tf.device('/gpu:1'),tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm1')

        # conv2
        with tf.device('/gpu:1'),tf.variable_scope('conv2') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3, 3, 32, 32],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[32],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')

        # pool2 and norm2
        with tf.device('/gpu:1'),tf.variable_scope('pooling2_lrn') as scope:

            pool2 = tf.nn.max_pool(conv2, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pooling2')
            norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm2')
                              
                              
                              
        # conv3
        with tf.device('/gpu:1'),tf.variable_scope('conv3') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3, 3, 32, 32],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[32],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm2, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name='conv3')

        # pool2 and norm2
        with tf.device('/gpu:1'),tf.variable_scope('pooling2_lrn') as scope:

            pool3 = tf.nn.max_pool(conv3, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pooling2')
            norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm2')                      

            # conv2
        # with tf.device('/gpu:1'), tf.variable_scope('conv3') as scope:
        #     weights = tf.get_variable('weights',
        #                               shape=[5, 5, 128, 256],
        #                               dtype=tf.float32,
        #                               initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        #     biases = tf.get_variable('biases',
        #                              shape=[128],
        #                              dtype=tf.float32,
        #                              initializer=tf.constant_initializer(0.1))
        #     conv = tf.nn.conv2d(norm2, weights, strides=[1, 1, 1, 1], padding='SAME')
        #     pre_activation = tf.nn.bias_add(conv, biases)
        #     conv2 = tf.nn.relu(pre_activation, name='conv2')
        #
        # # pool2 and norm2
        # with tf.device('/gpu:1'), tf.variable_scope('pooling3_lrn') as scope:
        #     pool2 = tf.nn.max_pool(conv2, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1],
        #                            padding='SAME', name='pooling2')
        #     norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
        #                       beta=0.75, name='norm2')
            # norm2 = tf.reshape(norm2, shape=[batch_size, -1])

        # with tf.device('/gpu:0'), tf.name_scope("dropout"):
        #     h_drop = tf.nn.dropout(norm2, dropout_keep_prob)

        # local3
        with tf.device('/gpu:1'),tf.variable_scope('local4') as scope:
            reshape = tf.reshape(norm3, shape=[batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.get_variable('weights',
                                      shape=[dim, 256],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[256],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        #
        # with tf.device('/gpu:0'),tf.name_scope("dropout"):
        #     h_drop = tf.nn.dropout(local3, dropout_keep_prob)

            # local4
        with tf.device('/gpu:1'),tf.variable_scope('local5') as scope:
            weights = tf.get_variable('weights',
                                      shape=[256, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[64],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

        # softmax
        with tf.device('/gpu:1'),tf.variable_scope('softmax_linear') as scope:
            weights = tf.get_variable('softmax_linear',
                                      shape=[64, n_classes],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[n_classes],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            # l2_loss += tf.nn.l2_loss(weights)
            # l2_loss += tf.nn.l2_loss(biases)
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

        return softmax_linear

    def prediction(self, output_layer):
        with tf.device('/gpu:1'),tf.name_scope("prediction"):
            score = output_layer
            pred_Y = tf.argmax(score, 1, name='pred_Y')
        return pred_Y

    # %%
    def losses(self, logits, labels):
        '''Compute loss from logits and labels
        Args:
            logits: logits tensor, float, [batch_size, n_classes]
            labels: label tensor, tf.int32, [batch_size]

        Returns:
            loss tensor of float type
        '''
        with tf.device('/gpu:1'),tf.variable_scope('loss') as scope:
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4),
                                                         tf.trainable_variables())
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
                (logits=logits, labels=labels, name='xentropy_per_example')
            loss = tf.reduce_mean(cross_entropy, name='loss') + reg
        return loss

    # %%
    def trainning(self, loss, learning_rate, global_step):
        '''Training ops, the Op returned by this function is what must be passed to
            'sess.run()' call to cause the model to train.

        Args:
            loss: loss tensor, from losses()

        Returns:
            train_op: The op for trainning
        '''
        with tf.device('/gpu:1'),tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    # %%
    def evaluation(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(logits, labels, 1)
            correct = tf.cast(correct, tf.float16)
            accuracy = tf.reduce_mean(correct)
            # tf.summary.scalar(scope.name + '/accuracy', accuracy)
        return accuracy, labels