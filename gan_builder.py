from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
import tensorflow as tf

def build(generator, discriminator):
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, label_smoothing=0.0, reduction="auto", name="binary_crossentropy"
    )
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.0004, decay=1e-10, clipvalue=1.0),
                  metrics=['accuracy'])
    model.summary()
    return model