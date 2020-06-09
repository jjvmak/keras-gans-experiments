from keras.models import Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout
from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU


def build(z_dim=100, depth=32, feature_map_dim=7, p=0.2):
    """ Builds the generator model.

        Args:
            z_dim: noise vector dimension
            depth: layer depth parameter
            feature_map_dim: layer feature map dimension parameter
            p: dropout rate parameter

        Returns:
            model: Keras model

    """

    # Define inputs.
    inputs = Input((z_dim,), name='input')

    # First dense layer.
    dense1 = Dense(feature_map_dim * feature_map_dim * depth, activation=None, name='dense_1_1')(inputs)
    dense1 = BatchNormalization(momentum=0.9, name='dense_1_2')(dense1)
    dense1 = LeakyReLU(0.2, name='dense_1_3')(dense1)

    # Reshape the output to match deconv layers.
    dense1 = Reshape((feature_map_dim, feature_map_dim, depth), name='dense_1_4')(dense1)
    dense1 = Dropout(p, name='dense_1_5')(dense1)

    # Deconv layer 1.
    # Since the inverse operation of the traditional convolution layer is needed,
    # the UpSampling2D layer is used to repeat the rows and columns of the input.
    # After the upsampling, the Conv2DTranspose layer is used to perform deconvolution.
    # I.e. instead of mapping the pixels to features, this block maps features to pixels.
    conv1 = UpSampling2D(name='conv1_1')(dense1)
    conv1 = Conv2DTranspose(int(depth / 2), kernel_size=3, padding='same', activation=None, name='conv1_2')(conv1)
    conv1 = BatchNormalization(momentum=0.9, name='conv1_3')(conv1)
    conv1 = LeakyReLU(0.2, name='conv1_4')(conv1)

    # Deconv layer 2.
    conv2 = UpSampling2D(name='conv2_1')(conv1)
    conv2 = Conv2DTranspose(int(depth / 4), kernel_size=3, padding='same', activation=None, name='conv2_2')(conv2)
    conv2 = BatchNormalization(momentum=0.9, name='conv2_3')(conv2)
    conv2 = LeakyReLU(0.2, name='conv2_4')(conv2)

    # Output layer.
    output = Conv2D(1, kernel_size=3, padding='same', name='output1_1')(conv2)
    output = Activation('sigmoid', name='output1_2')(output)

    # Model definition.
    model = Model(inputs=inputs, outputs=output, name='generator')
    model.summary()

    return model
