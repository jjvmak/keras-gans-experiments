import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from keras.datasets import mnist
import generator_builder
import discriminator_builder
import gan_builder



def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def train(epochs=2000, batch=128):
    d_metrics = []
    a_metrics = []

    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0

    for i in range(epochs):

        if i % 100 == 0:
            print(i)

        real_imgs = np.reshape(data[np.random.choice(data.shape[0], batch, replace=False)], (batch, 28, 28, 1))
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

        if (i + 1) % 100 == 0:

            print('Epoch #{}'.format(i + 1))
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, running_d_loss / i, running_d_acc / i)
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, running_a_loss / i, running_a_acc / i)
            print(log_mesg)

            gen_imgs = generator.predict(static_noise)

            plt.figure(figsize=(5, 5))

            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k + 1)
                plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
                plt.axis('off')

            plt.tight_layout()
            plt.show()

    return a_metrics, d_metrics


(data, y_train), (x_test, y_test) = mnist.load_data()
data = np.reshape(data, (data.shape[0], 28, 28, 1))
img_w, img_h = data.shape[1:3]

# build discriminator
discriminator = discriminator_builder.build()
discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.008, decay=6e-8, clipvalue=1.0),
                      metrics=['accuracy'])

# build generator
generator = generator_builder.build()

# build gan
gan = gan_builder.build(generator, discriminator)

static_noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
a_metrics_complete, d_metrics_complete = train(epochs=3000, batch=50)
