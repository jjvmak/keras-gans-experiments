import numpy as np
import config
from keras.datasets import mnist
import os
import generator_builder as gen
import discriminator_builder
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import cv2

# np.random.seed(10)

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def show_images(C, noise):
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))

    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i + 1)
        if C.channels == 1:
            plt.imshow(image.reshape((C.img_rows, C.img_cols)), cmap='gray')
        else:
            plt.imshow(image.reshape((C.img_rows, C.img_cols, C.channels)))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def save_images(C ,noise):
    generated_images = generator.predict(noise)

    for i, image in enumerate(generated_images):

        if C.channels == 1:
            image = image.reshape(C.img_rows, C.img_cols)
            image *= 127
            cv2.imwrite('./fcgan-images/{}.png'.format(i), image)
        else:
           image = image.reshape((C.img_rows, C.img_cols, C.channels))
           image *= 127
           cv2.imwrite('./fcgan-images/{}.png'.format(i), image)


C = config.Config()
generator = gen.create_generator(C)
generator.load_weights(C.generator_model_path, by_name=True)
noise = np.random.normal(0, 1, size=(100, C.noise_dim))
show_images(C, noise)