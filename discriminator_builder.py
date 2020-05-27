from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, Activation

def build(w=28, h=28, depth=64, p=0.4):
    # Define inputs
    inputs = Input((w, h, 1))

    # conv laters
    conv1 = Conv2D(depth * 1, 5, strides=2, padding='same', activation='relu')(inputs)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(depth * 2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(depth * 4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(depth * 8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))

    # output layer
    output = Dense(1)(conv4)  # , activation='sigmoid')(conv4)
    output = Activation('sigmoid')(output)

    # model def
    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model