from keras.models import Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten
from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D
from keras.optimizers import RMSprop


def build(z_dim=100, depth=64, p=0.4):  # 100 dimmensional latent space / p represents dropout rate
    # define inputs
    inputs = Input((z_dim,))

    # first dense layer
    dense1 = Dense(7 * 7 * 64)(inputs)
    dense1 = BatchNormalization(momentum=0.9)(dense1)  # Helps maintain the mean and variabnce of the params
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7, 7, 64))(dense1)
    dense1 = Dropout(p)(dense1)

    # deconv layers
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth / 2), kernel_size=5, padding='same', activation=None)(conv1)
    conv1 = BatchNormalization(momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth / 4), kernel_size=5, padding='same', activation=None)(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)

    conv3 = Conv2DTranspose(int(depth / 8), kernel_size=5, padding='same', activation=None)(conv2)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)

    # output layer
    output = Conv2D(1, kernel_size=5, padding='same')(conv3)  # , activation='sigmoid')(conv3)
    output = Activation('tanh')(output)

    # model definition
    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model