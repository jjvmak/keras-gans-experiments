from keras.models import Sequential
from keras.optimizers import RMSprop

# Combined GAN model.
def build(generator, discriminator):
    # This results in a model that takes noise vector as input,
    # generates image and outputs a value,
    # how convinced the discriminator is that the image is real.
    # I.e. D(G(z))
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.0004, decay=1e-10, clipvalue=1.0),
                  metrics=['accuracy'])
    model.summary()
    return model