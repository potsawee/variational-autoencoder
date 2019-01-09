import numpy as np
import tensorflow as tf


#  Building blocks for the neural network of the variational autoencoder
def conv_layer(input, filter_size, num_filters, stride, name):
    with tf.variable_scope(name):
        # shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, input.get_shape()[-1], num_filters]
        weights = tf.get_variable('w', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
        biases = tf.get_variable('b', shape=[num_filters], initializer=tf.constant_initializer(0.0))
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, stride, stride, 1], padding='SAME')
        layer += biases
        return layer

def conv_transpose_layer(input, output_shape, ksize, stride, name):
    with tf.variable_scope(name):
        # h, w, out, in
        filter = tf.get_variable('w', [ksize, ksize, output_shape[-1], input.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.05))
        # layer = tf.nn.conv2d_transpose(input, filter, output_shape=output_shape, strides=[1,stride,stride,1])
        layer = tf.nn.conv2d_transpose(input, filter, output_shape=output_shape, strides=[1,stride,stride,1])

        return layer

def relu_layer(input, name):
    with tf.variable_scope(name):
        layer = tf.nn.relu(input)
        return layer

def pooling_layer(input, ksize, stride, name):
    with tf.variable_scope(name):
        layer = tf.nn.max_pool(value=input, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')
        return layer

def fc_layer(input, num_inputs, num_outputs, name):
    with tf.variable_scope(name):
        weights = tf.get_variable('w', shape=[num_inputs, num_outputs], initializer=tf.truncated_normal_initializer(stddev=0.05))
        biases = tf.get_variable('b', shape=[num_outputs], initializer=tf.constant_initializer(0.0))
        layer = tf.matmul(input, weights) + biases
        return layer
