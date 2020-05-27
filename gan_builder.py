from keras.models import Sequential
from keras.optimizers import RMSprop

def build(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.004, decay=1e-10, clipvalue=1.0),
                  metrics=['accuracy'])
    model.summary()
    return model