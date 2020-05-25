import numpy as np
import config
from keras.datasets import mnist
import os
import generator
import discriminator
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import cv2

# TODO separate the training and testing modules
# TODO verbose training
# TODO TensorBoard logging

def save_images(C ,noise):
    generated_images = generator.predict(noise)

    for i, image in enumerate(generated_images):

        if C.channels == 1:
            image = image.reshape(C.img_rows, C.img_cols)
            image *= 255
            cv2.imwrite('./fcgan-images/{}.png'.format(i), image)
        else:
           image = image.reshape((C.img_rows, C.img_cols, C.channels))
           image *= 255
           cv2.imwrite('./fcgan-images/{}.png'.format(i), image)


np.random.seed(10)
C = config.Config()

# load the mnist data and flatten the input data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32)) / 255
x_train = x_train.reshape(-1, C.img_rows*C.img_cols*C.channels)

# create path for images
if not os.path.isdir(C.save_path):
    os.mkdir(C.save_path)

generator = generator.create_generator(C)
discriminator = discriminator.create_discriminator(C)

''' 
Set the trainable false, since
generator and discriminator are combined into single model.
This allows the generator to understand the discriminator 
so it can update itself more effectively. 
'''
discriminator.trainable = False

# call compile after setting trainable = False
generator.compile(loss='binary_crossentropy', optimizer=C.optimizer)
discriminator.compile(loss='binary_crossentropy', optimizer=C.optimizer)

gan_input = Input(shape=(C.noise_dim,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=C.optimizer)

# training
for epoch in range(C.epochs):
    for batch in range(C.steps_per_epoch):
        # generate random noise
        noise = np.random.normal(0, 1, size=(C.batch_size, C.noise_dim))

        # generate fake data
        fake_x = generator.predict(noise)

        # draw real sample from dataset
        real_x = x_train[np.random.randint(0, x_train.shape[0], size=C.batch_size)]

        x = np.concatenate((real_x, fake_x))

        disc_y = np.zeros(2 * C.batch_size)
        # label smoothing
        # Salimans et al. 2016
        disc_y[:C.batch_size] = 0.9

        d_loss = discriminator.train_on_batch(x, disc_y)

        y_gen = np.ones(C.batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')


noise = np.random.normal(0, 1, size=(100, C.noise_dim))
save_images(C, noise)






