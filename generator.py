from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU


def create_generator(C):
    generator = Sequential()

    generator.add(Dense(256, input_dim=C.noise_dim))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(C.img_rows * C.img_cols * C.channels, activation='tanh'))

    # don't compile here
    # generator.compile(loss='binary_crossentropy', optimizer=C.optimizer)
    return generator