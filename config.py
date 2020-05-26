from keras.optimizers import Adam
class Config:
    # specifies the length of the random noise vector, z
    noise_dim = 100
    batch_size = 16
    steps_per_epoch = 3750
    epochs = 1
    save_path = 'fcgan-images'
    img_rows = 28
    img_cols = 28
    channels = 1
    optimizer = Adam(lr=1e-5)
    model_path = 'model_fcgan.hdf5'