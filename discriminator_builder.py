from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, Activation, MaxPooling2D
from keras.layers import LeakyReLU

def build(w=28, h=28, depth=32, p=0.2):
    # Define inputs
    inputs = Input((w, h, 1))

    # conv laters
    conv1 = Conv2D(depth * 1,(3,3), padding='same', activation=None)(inputs)
    conv1 = LeakyReLU(0.2)(conv1)
    conv1 = Conv2D(depth * 1,(3,3), padding='same', activation=None)(conv1)
    conv1 = LeakyReLU(0.2)(conv1)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(depth * 2,(3,3), padding='same', activation=None)(conv1)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = Conv2D(depth * 2,(3,3), padding='same', activation=None)(conv2)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(depth * 4,(1,1), padding='same', activation=None)(conv2)
    conv3 = LeakyReLU(0.2)(conv3)
    conv3 = Dropout(p)(conv3)

    #conv4 = Conv2D(depth * 8,(1,1), padding='same', activation='relu')(conv3)
    # conv4 = Conv2D(depth * 8,(1,1), padding='same', activation='relu')(conv4)
    conv4 = Flatten()(Dropout(p)(conv3))

    # output layer
    output = Dense(1)(conv4)
    output = Activation('sigmoid')(output)

    # model def
    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model