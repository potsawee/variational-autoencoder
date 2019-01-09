import os
import numpy as np
import tensorflow as tf
from PIL import Image


# Building blocks for the neural network of the variational autoencoder
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


# Data processing
def load_data(path):
    # data from Kaggle - Digit Recogniser
    with open(path, 'r') as file:
        lines = file.readlines()
    num = len(lines) - 1
    x = np.zeros(shape=(num,784))
    # y = np.zeros(shape=(num))

    lines = lines[1:] # ignore pixel0, pixel1,...
    np.random.shuffle(lines)

    for i, line in enumerate(lines[1:]):
        items = line.strip().split(',')
        # y[i] = int(items[0])
        for j, val in enumerate(items[1:]):
            x[i,j] = int(val) / 255 # normalise
    return x

def merge(images, dimension=28):
    new_im = Image.new(mode='RGB',size=(8*dimension, 8*dimension))
    x_offset = 0
    y_offset = 0
    for i in range(images.shape[0]):
        im = images[i,:,:,:]
        im = Image.fromarray(im, mode='RGB')
        new_im.paste(im, (x_offset,y_offset))
        x_offset += dimension
        ii = i+1
        if ii % 8 == 0:
            x_offset = 0
            y_offset += dimension
    return new_im

def load_dog_images(dir, dimension=200):
    files = os.listdir(dir)
    num = len(files)
    images = np.zeros((num, dimension, dimension, 3))
    for i, file in enumerate(files):
        path = dir + file
        img = Image.open(path)
        img = img.resize((dimension, dimension), Image.BILINEAR)
        img = np.array(img) / 255
        images[i,:,:,:] = img
    return images
