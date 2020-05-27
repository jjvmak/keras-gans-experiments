import numpy as np
import config
from keras.datasets import mnist
from keras.models import Sequential, Model
import os
import generator_builder as gen
import discriminator_builder as disc
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
from keras.utils import generic_utils

#np.random.seed(10)

# TODO TensorBoard logging

def show_images(C, noise, epoch=None):
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))

    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i + 1)
        if C.channels == 1:
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{C.save_path}/gan-images_epoch-{epoch}.png')

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

C = config.Config()

# load the mnist data and flatten the input data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32))
x_train = x_train.reshape(-1, C.img_rows*C.img_cols*C.channels)

# create path for images
if not os.path.isdir(C.save_path):
    os.mkdir(C.save_path)

generator = gen.create_generator(C)
discriminator = disc.create_discriminator(C)
discriminator.compile(loss='binary_crossentropy', optimizer=C.optimizer)

# TODO separate
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
gan.compile(loss='binary_crossentropy',
            optimizer=C.optimizer,
            metrics=['accuracy'])
gan.summary()



# training
static_noise = np.random.normal(0, 1, size=(100, C.noise_dim))
losses = np.zeros((C.steps_per_epoch, 1))

for epoch in range(C.epochs):
    iter_num = 0
    progbar = generic_utils.Progbar(C.steps_per_epoch)
    for batch in range(C.steps_per_epoch):

        # generate random noise
        noise = np.random.normal(0, 1, size=(C.batch_size, C.noise_dim))

        # generate fake data
        fake_x = generator.predict(noise)

        # draw real sample from dataset
        real_x = np.reshape(x_train[np.random.randint(0, x_train.shape[0], size=C.batch_size)], (C.batch_size, 28, 28, 1))

        x = np.concatenate((real_x, fake_x))

        disc_y = np.zeros(2 * C.batch_size)
        # label smoothing
        # Salimans et al. 2016
        disc_y[:C.batch_size] = 1.0

        make_trainable(discriminator, True)
        d_loss = discriminator.train_on_batch(x, disc_y)

        make_trainable(discriminator, False)
        y_gen = np.ones(C.batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)

        #losses[iter_num] = g_loss
        #progbar.update(iter_num, [('g_loss', losses[iter_num])])
        iter_num += 1
    print('\nend of epoch')
    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')

    show_images(C, static_noise, epoch)

gan.save_weights(C.gan_model_path)
discriminator.save_weights(C.discriminator_model_path)
generator.save_weights(C.generator_model_path)









