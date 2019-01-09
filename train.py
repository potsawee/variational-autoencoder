import os
import sys
import random
import numpy as np
import tensorflow as tf
import pdb
from model import VariationalAutoEncoder
from helper import *

def get_batches(x, batch_size, dimension=28):
    num = x.shape[0]

    # for mnist data
    # _x = np.reshape(x, (num, dimension, dimension, 1))

    _x = x

    batches = []

    for i in range(int(num/batch_size)):
        i_start = i * batch_size
        i_end = i_start + batch_size
        batch = _x[i_start:i_end, :, :, :]
        batches.append(batch)

    return batches


def train():

    # ------------------------------ setting ------------------------------ #
    if 'X_SGE_CUDA_DEVICE' in os.environ:
        print('running on the stack...')
        cuda_device = os.environ['X_SGE_CUDA_DEVICE']
        print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    else: # development only e.g. air202
        print('running locally...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '1' # choose the device (GPU) here

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True # Whether the GPU memory usage can grow dynamically.
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95 # The fraction of GPU memory that the process can use.
    # --------------------------------------------------------------------- #

    # MNIST
    # data_path = '/home/alta/BLTSpeaking/ged-pm574/my-projects/mnist/data/train.csv'
    # x_train = load_data(data_path)

    # DOGS
    data_train = 'data/dogs/train/'
    data_test = 'data/dogs/test/'
    x_train = load_dog_images(data_train)
    print('load train done...')
    # x2 = load_dog_images(data_test)
    # print('load test done...')
    # x_train = np.concatenate((x_train,x2), axis=0)

    save_path = 'save/dogs2/vae-v1'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vae = VariationalAutoEncoder()
    vae.build_network()
    vae.build_loss_function()
    vae.build_optimiser()
    vae.build_generator()

    saver = tf.train.Saver(max_to_keep=1)

    batch_size = 512
    num_epochs = 2000

    batches = get_batches(x_train, batch_size)

    my_z_gen = np.random.normal(0.0,1.0,size=(64,vae.z_size))


    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):

            random.shuffle(batches)
            for i, batch in enumerate(batches):
                feed_dict = {vae.inputs: batch}
                _, loss = sess.run([vae.train_op, vae.loss], feed_dict=feed_dict)
                if i == 0:
                    print("epoch: {} --- loss: {:.5f}".format(epoch, loss))

            # print("################## EPOCH {} done ##################".format(epoch))
            if epoch % 2 == 0:
                saver.save(sess, save_path + '/model', global_step=epoch)
                # generate image
                feed_dict = {vae.z_gen: my_z_gen}
                [output_gen] = sess.run([vae.output_gen], feed_dict=feed_dict)
                output_gen = np.multiply(255, output_gen)
                output_gen = np.array(output_gen, dtype=float)
                result_name = 'results/dogs2/gen-' + str(epoch) + '.jpg'
                new_img = merge(output_gen, dimension=200)
                new_img.save(result_name)


def main():
    train()

if __name__ == '__main__':
    main()
