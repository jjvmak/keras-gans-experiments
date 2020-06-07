import numpy as np
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from keras.datasets import mnist
import generator_builder
import discriminator_builder
import gan_builder
import data_handler
import data_handler_custom
from datetime import datetime
import os
from optparse import OptionParser
import pandas as pd


# Iterates through network layers and sets layer.trainable true / false.
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


# Training loop.
def train(epochs=1000, batch=150):
    # Arrays for storing the discriminant and generator accuracy and loss metrics for each epoch.
    d_metrics = []
    a_metrics = []

    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0

    for i in range(epochs):

        # Draws a batch of random samples from real images data.
        real_imgs = data_sampler.random_sample(batch)
        # Generates fake images from uniform noise.
        fake_imgs = generator.predict(np.random.uniform(-1.0, 1.0, size=[batch, 100]))

        # Combines real images and fake images into x array.
        # Creates corresponding labels.
        # Label smoothing is set to 0.9
        x = np.concatenate((real_imgs, fake_imgs))
        y = np.zeros(2 * batch)
        y[:batch] = 0.9

        # Allow discriminator training.
        make_trainable(discriminator, True)

        # Trains the discriminator with x and y.
        # Saves the metrics into arrays.
        d_metrics.append(discriminator.train_on_batch(x, y))
        running_d_loss += d_metrics[-1][0]
        running_d_acc += d_metrics[-1][1]

        # Set the discriminator not trainable.
        make_trainable(discriminator, False)

        # Generates uniform noise.
        noise = np.random.uniform(-1.0, 1.0, size=[batch, 100])
        y = np.ones([batch, 1])

        # Trains the combined GANs model with generated noise.
        # Saves the metrics into arrays.
        a_metrics.append(gan.train_on_batch(noise, y))
        running_a_loss += a_metrics[-1][0]
        running_a_acc += a_metrics[-1][1]

        # Logs the accuracy and loss metrics.
        if (i + 1) % int(options.log_frequency) == 0:
            print('Epoch #{}'.format(i + 1))
            log_mesg = "[D loss: %f, acc: %f]" % (running_d_loss / i, running_d_acc / i)
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, running_a_loss / i, running_a_acc / i)
            print(log_mesg)

        # Generate images with the current generator model.
        # Creates 4x4 figure and saves it disk.
        if (i + 1) % int(options.generator_frequency) == 0:
            gen_imgs = generator.predict(static_noise)
            plt.figure(figsize=(5, 5))

            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k + 1)
                plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f'{save_path}/gan-images_epoch_{i + 1}.png')

    return a_metrics, d_metrics


# parse options
parser = OptionParser()
parser.add_option('-e', '--epochs', dest='epochs', default=2000, help='Number of epochs.')
parser.add_option('-b', '--batch', dest='batch', default=30,
                  help='Batch size. Use only dozens of equality. For example 50.')
parser.add_option('-l', '--log-frequency', dest='log_frequency', default=50, help='Set training logging frequency.')
parser.add_option('-g', '--generator-frequency', dest='generator_frequency', default=50,
                  help='Set generator frequency.')
parser.add_option('-d', '--dataset', dest='dataset', default='custom',
                  help='Set dataset.')
(options, args) = parser.parse_args()

# Creates a data handler that currently is implemented to work with LFW dataset.
# Image width and height is set to 64x64.
# For practical reasons, feature_map_dim and depth are initialized see:
# generator_builder.build()
if options.dataset == 'custom':
    data_sampler = data_handler_custom.Data_handler_custom()
    img_w, img_h = 64, 64
    feature_map_dim = 16
    depth = 32

# Creates a data handler for MNIST dataset.
# Reshapes the data to (n, 28, 28, 1)
# Normalize values to 0 - 1 range.
else:
    (data, y_train), (x_test, y_test) = mnist.load_data()
    data = np.reshape(data, (data.shape[0], 28, 28, 1))
    data = data / 255
    data_sampler = data_handler.Data_handler(data, y_train)
    img_w, img_h = data.shape[1:3]
    feature_map_dim = 7
    depth = 64

# Build discriminator with binary cross entropy and RMSprop.
discriminator = discriminator_builder.build(w=img_w, h=img_h, depth=depth)
discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0008, decay=6e-8, clipvalue=1.0),
                      metrics=['accuracy'])

# Build generator.
generator = generator_builder.build(depth=depth, feature_map_dim=feature_map_dim)

# Combines the models into one GANs model.
# This results in a model that takes noise vector as input,
# generates image and outputs a value,
# how convinced the discriminator is that the image is real.
# I.e. D(G(z))
gan = gan_builder.build(generator, discriminator)

# Static noise for  getting consistent results.
static_noise = np.random.uniform(-1.0, 1.0, size=[16, 100])

# Save path for generated results.
save_path = "./results/" + datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# Starts the training loop and saves the metrics.
a_metrics_complete, d_metrics_complete = train(epochs=int(options.epochs), batch=int(options.batch))

# Saves the generator weights after the training is completed.
print('saving generator weights')
generator.save_weights('generator_model.hdf5')

# Creates figures of the metrics and saves them to the disk.
try:
    ax = pd.DataFrame({
        'Generator': [metric[0] for metric in a_metrics_complete],
        'Discriminator': [metric[0] for metric in d_metrics_complete],
    }).plot(title='Training Loss', logy=False)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.get_figure().savefig(f'{save_path}/loss.png')

    ax = pd.DataFrame({
        'Generator': [metric[1] for metric in a_metrics_complete],
        'Discriminator': [metric[1] for metric in d_metrics_complete],
    }).plot(title='Training Accuracy', logy=False)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.get_figure().savefig(f'{save_path}/accuracy.png')

except Exception as e:
    print(e)
