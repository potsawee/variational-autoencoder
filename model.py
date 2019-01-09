import tensorflow as tf
from helper import *

class VariationalAutoEncoder(object):
    def __init__(self):
        self.z_size = 2
        self.learning_rate = 0.00005

    def build_network(self):
        # this inputs must be normalised so that 0.0 to 1.0
        self.inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name="inputs")

        with tf.variable_scope("encoder"):
            # CNN layer 1
            conv1 = conv_layer(self.inputs, filter_size=5, num_filters=6, stride=2, name="conv1")
            relu1 = relu_layer(conv1, name="relu1")
            pool1 = pooling_layer(relu1, ksize=2, stride=2, name="pool1")

            # CNN layer 2
            conv2 = conv_layer(pool1, filter_size=5, num_filters=16, stride=2, name="conv2")
            relu2 = relu_layer(conv2, name="relu2")
            pool2 = pooling_layer(relu2, ksize=2, stride=2, name="pool2")

            # Flatten layer
            self.num_flatten = pool2.get_shape()[1:4].num_elements()
            # note: num_elements returns the total number of elements
            flatten = tf.reshape(pool2, [-1, self.num_flatten])

            # Fully connected layer
            fc1 = fc_layer(flatten, num_inputs=self.num_flatten, num_outputs=128, name="fc1")
            relu_fc1 = relu_layer(fc1, name="relu_fc1")

            # Latent Space
            self.z_mean = fc_layer(relu_fc1, num_inputs=128, num_outputs=self.z_size, name="z_mean")
            self.z_std = fc_layer(relu_fc1, num_inputs=128, num_outputs=self.z_size, name="z_std")

        with tf.variable_scope("decoder"):
            # Generate sample from z_mean and z_std
            s = tf.shape(self.inputs)
            samples = tf.random_normal([s[0], self.z_size], 0, 1, dtype=tf.float32)
            gaussian_z = self.z_mean + samples*self.z_std

            # Fully connected layer
            fc1 = fc_layer(gaussian_z, num_inputs=self.z_size, num_outputs=128, name="fc1")
            relu_fc1 = relu_layer(fc1, name="relu_fc1")

            fc2 = fc_layer(relu_fc1, num_inputs=128, num_outputs=7*7*32, name="fc2")
            relu_fc2 = relu_layer(fc2, name="relu_fc2")

            # Reshape to be [None, height, width, depth]
            hidden = tf.reshape(relu_fc2, shape=[s[0], 7, 7, 32])

            # Deconvolution
            convt1 = conv_transpose_layer(hidden, output_shape=[s[0], 14, 14, 16], ksize=5, stride=2, name="convt1")
            relu_convt1 = relu_layer(convt1, name="relu_convt1")

            convt2 = conv_transpose_layer(relu_convt1, output_shape=[s[0], 28, 28, 1], ksize=5, stride=2, name="convt2")

            self.output = tf.nn.sigmoid(convt2)

    def build_loss_function(self):
        # reconstruction loss E[log p(x|z)]
        s = tf.shape(self.inputs)
        _input = tf.reshape(self.inputs, shape=[s[0], 28*28])
        _output = tf.reshape(self.output, shape=[s[0], 28*28])
        self.loss_r = _input*tf.log(1e-8+_output) + (1-_input)*tf.log(1e-8+1-_output)
        self.loss_r = tf.reduce_sum(self.loss_r,axis=1)
        self.loss_r = -1.0 * tf.reduce_mean(self.loss_r)

        # KL-divergence loss (latent loss) - 4F10: handout 10, page 7
        self.loss_kl = tf.square(self.z_mean) + tf.square(self.z_std) - tf.log(tf.square(self.z_std)) - 1
        self.loss_kl = tf.reduce_sum(self.loss_kl)
        self.loss_kl = 0.5 * tf.reduce_mean(self.loss_kl)

        self.loss = self.loss_r + self.loss_kl

    def build_optimiser(self):
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params))
