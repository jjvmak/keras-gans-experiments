import numpy as np
import config
import generator_builder
import matplotlib.pyplot as plt
from datetime import datetime
import os

def generate(noise, i=0, save=True):
    gen_imgs = generator.predict(noise)

    plt.figure(figsize=(5, 5))

    for k in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, k + 1)
        plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}/generated_images_{i}.png')
    else:
        plt.show()


save_path = "./generated/" + datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.isdir(save_path):
    os.mkdir(save_path)

C = config.Config()
generator = generator_builder.build()
generator.load_weights('generator_model.hdf5', by_name=True)

for i in range(10):
    noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
    generate(noise=noise, i=i, save=True)