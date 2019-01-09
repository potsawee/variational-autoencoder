import os
import sys
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import pdb
from model import VariationalAutoEncoder

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

def get_batches(x, batch_size):
    num = x.shape[0]
    _x = np.reshape(x, (num, 28, 28, 1))

    batches = []

    for i in range(int(num/batch_size)):
        i_start = i * batch_size
        i_end = i_start + batch_size
        batch = _x[i_start:i_end, :, :, :]
        batches.append(batch)

    return batches


def merge(images):
    new_im = Image.new(mode='L',size=(8*28, 8*28))
    x_offset = 0
    y_offset = 0
    for i in range(images.shape[0]):
        im = images[i,:,:,0]
        im = Image.fromarray(im)
        new_im.paste(im, (x_offset,y_offset))
        x_offset += 28
        ii = i+1
        if ii % 8 == 0:
            x_offset = 0
            y_offset += 28
    return new_im


def train():

    # ------------------------------ setting ------------------------------ #
    if 'X_SGE_CUDA_DEVICE' in os.environ:
        print('running on the stack...')
        cuda_device = os.environ['X_SGE_CUDA_DEVICE']
        print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    else: # development only e.g. air202
        print('running locally...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '3' # choose the device (GPU) here

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True # Whether the GPU memory usage can grow dynamically.
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95 # The fraction of GPU memory that the process can use.
    # --------------------------------------------------------------------- #

    data_path = '/home/alta/BLTSpeaking/ged-pm574/my-projects/mnist/data/train.csv'
    x_train = load_data(data_path)

    save_path = 'save/vae-v1'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vae = VariationalAutoEncoder()
    vae.build_network()
    vae.build_loss_function()
    vae.build_optimiser()
    vae.build_generator()

    saver = tf.train.Saver(max_to_keep=1)

    batch_size = 1000
    num_epochs = 50

    batches = get_batches(x_train, batch_size)
    z1 = np.linspace(-1.2, 1.2, num=8)
    z2 = np.linspace(-1.2, 1.2, num=8)
    my_z_gen = np.zeros((64,2))
    k = 0
    for i in z1:
        for j in z2:
            my_z_gen[k, 0] = i
            my_z_gen[k, 1] = j
            k += 1

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
                result_name = 'results/gen-' + str(epoch) + '.jpg'
                new_img = merge(output_gen)
                new_img.save(result_name)


def main():
    train()

if __name__ == '__main__':
    main()
