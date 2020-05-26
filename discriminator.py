from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU

def create_discriminator(C):
    discriminator = Sequential()

    discriminator.add(Dense(1024, input_dim=C.img_rows * C.img_cols * C.channels, name='disc_1'))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(512, name='disc_2'))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(256, name='disc_3'))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='sigmoid', name='disc_4'))

    discriminator.compile(loss='binary_crossentropy', optimizer=C.optimizer)
    return discriminator