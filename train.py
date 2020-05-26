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

np.random.seed(10)

# TODO separate the training and testing modules
# TODO verbose training
# TODO TensorBoard logging

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

C = config.Config()

# load the mnist data and flatten the input data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
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
make_trainable(discriminator, False)
generator.compile(loss='binary_crossentropy', optimizer=C.optimizer)
discriminator.compile(loss='binary_crossentropy', optimizer=C.optimizer)

gan_input = Input(shape=(C.noise_dim,))
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=C.optimizer)


# pre train
noise = np.random.normal(0, 1, size=(C.batch_size, C.noise_dim))
generated_images = generator.predict(noise)
real_x = x_train[np.random.randint(0, x_train.shape[0], size=C.batch_size)]
x = np.concatenate((real_x, generated_images))
disc_y = np.zeros(2 * C.batch_size)
disc_y[:C.batch_size] = 1.0
make_trainable(discriminator, True)
discriminator.fit(x, disc_y, epochs=5, batch_size=8)


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
        disc_y[:C.batch_size] = 1.0

        make_trainable(discriminator, True)
        d_loss = discriminator.train_on_batch(x, disc_y)

        make_trainable(discriminator, False)
        y_gen = np.ones(C.batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')

gan.save_weights(C.model_path)









