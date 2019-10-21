class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):
        # Trainging
        self.n_class = 3
        self.image_w = 20
        self.image_h = 20
        self.batch_size = 50
        self.training_epoch = 25
        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.001
        self.batch_size_per_class = 40 # if equal sampling
        # dropout keep prob for fully connected layers
        self.dropout_keep_prob = 0.8
        # dropout keep prob for convolutional layers
        self.dropout_keep_prob_conv = 0.8
        self.evaluate_every = 100

