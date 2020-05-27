from keras.optimizers import Adam, RMSprop
class Config:
    # specifies the length of the random noise vector, z
    noise_dim = 100
    batch_size = 16
    steps_per_epoch = 200
    epochs = 10
    save_path = 'fcgan-images'
    img_rows = 28
    img_cols = 28
    channels = 1
    optimizer = RMSprop(lr=0.0004, decay=1e-10, clipvalue=1.0)
    gan_model_path = 'gan_model_fcgan.hdf5'
    discriminator_model_path = 'discriminator_model_fcgan.hdf5'
    generator_model_path = 'generator_model_fcgan.hdf5'
    pre_train = False