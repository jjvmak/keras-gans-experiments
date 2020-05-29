import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from matplotlib import pyplot as plt
from keras.datasets import mnist
import generator_builder
import discriminator_builder
import gan_builder
import tensorflow as tf
import data_handler
from datetime import datetime
import os
from optparse import OptionParser


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def train(epochs=1000, batch=150):

    d_metrics = []
    a_metrics = []

    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0

    for i in range(epochs):

        # real_imgs = np.reshape(data[np.random.choice(data.shape[0], batch, replace=False)], (batch, 28, 28, 1))
        real_imgs = data_sampler.random_sample(batch)
        fake_imgs = generator.predict(np.random.uniform(-1.0, 1.0, size=[batch, 100]))

        x = np.concatenate((real_imgs, fake_imgs))
        y = np.ones([2 * batch, 1])
        y[batch:, :] = 0

        make_trainable(discriminator, True)

        d_metrics.append(discriminator.train_on_batch(x, y))
        running_d_loss += d_metrics[-1][0]
        running_d_acc += d_metrics[-1][1]

        make_trainable(discriminator, False)

        noise = np.random.uniform(-1.0, 1.0, size=[batch, 100])
        y = np.ones([batch, 1])

        a_metrics.append(gan.train_on_batch(noise, y))
        running_a_loss += a_metrics[-1][0]
        running_a_acc += a_metrics[-1][1]

        if (i + 1) % int(options.log_frequency) == 0:

            print('Epoch #{}'.format(i + 1))
            log_mesg = "[D loss: %f, acc: %f]" % (running_d_loss / i, running_d_acc / i)
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, running_a_loss / i, running_a_acc / i)
            print(log_mesg)


        if (i + 1) % int(options.generator_frequency) == 0:

            gen_imgs = generator.predict(static_noise)
            plt.figure(figsize=(5, 5))

            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k + 1)
                plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
                plt.axis('off')

            plt.tight_layout()
            # plt.show()
            plt.savefig(f'{save_path}/gan-images_epoch_{i + 1}.png')

    return a_metrics, d_metrics

# parse options
parser = OptionParser()
parser.add_option('-e', '--epochs', dest='epochs', default=1000, help='Number of epochs.')
parser.add_option('-b', '--batch', dest='batch', default=150, help='Batch size. Use only dozens of equality. For example 50.')
parser.add_option('-l', '--log-frequency', dest='log_frequency', default=50, help='Set training logging frequency.')
parser.add_option('-g', '--generator-frequency', dest='generator_frequency', default=50, help='Set generator frequency.')
(options, args) = parser.parse_args()

(data, y_train), (x_test, y_test) = mnist.load_data()
data = np.reshape(data, (data.shape[0], 28, 28, 1))
data = data / 255
data_sampler = data_handler.Data_handler(data, y_train)
img_w, img_h = data.shape[1:3]

# build discriminator
loss = tf.keras.losses.BinaryCrossentropy(
    from_logits=False, label_smoothing=0.1, reduction="auto", name="binary_crossentropy"
)
discriminator = discriminator_builder.build()
discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0008, decay=6e-8, clipvalue=1.0),
                      metrics=['accuracy'])

# build generator
generator = generator_builder.build()

# build gan
gan = gan_builder.build(generator, discriminator)
static_noise = np.random.uniform(-1.0, 1.0, size=[16, 100])

save_path = "./results/" + datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.isdir(save_path):
    os.mkdir(save_path)

a_metrics_complete, d_metrics_complete = train(epochs=int(options.epochs), batch=int(options.batch))

print('saving generator weights')
generator.save_weights('generator_model.hdf5')
