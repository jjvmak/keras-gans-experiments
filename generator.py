from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU


def create_generator(C):
    generator = Sequential()

    generator.add(Dense(256, input_dim=C.noise_dim, name='gen_1'))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512, name='disc_2'))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024, name='disc_3'))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(C.img_rows * C.img_cols * C.channels, activation='tanh', name='disc_4'))

    generator.compile(loss='binary_crossentropy', optimizer=C.optimizer)
    return generator