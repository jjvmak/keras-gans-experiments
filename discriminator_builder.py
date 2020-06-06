from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import LeakyReLU, Convolution2D, Add, MaxPooling2D


def build(w=28, h=28, depth=32, p=0.2):
    # Define inputs
    inputs = Input((w, h, 1))

    # conv laters
    x = Conv2D(depth * 1, (3, 3), padding='same', activation=None)(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(depth * 1, (3, 3), padding='same', activation=None)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(depth * 2, (3, 3), padding='same', activation=None)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(depth * 2, (3, 3), padding='same', activation=None)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(depth * 4, (1, 1), padding='same', activation=None)(x)
    x = LeakyReLU(0.2)(x)

    conv4 = Flatten()(Dropout(p)(x))

    # output layer
    output = Dense(1)(conv4)
    output = Activation('sigmoid')(output)

    # model def
    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model