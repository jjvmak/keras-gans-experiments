import os
import cv2
import numpy as np


class Data_handler_lfw:

    def __init__(self):
        self.arr = os.listdir(path='./lfw-deepfunneled-all')
        self.arr_len = len(self.arr)
        self.random_sample(10)

    def read_image(self, name):
        img = cv2.imread('./lfw-deepfunneled-all/' + name)
        # img = img / 255
        resized = cv2.resize(img, (int(28), int(28)))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = gray / 255
        gray_reshaped = np.reshape(gray, (28, 28, 1))
        return gray_reshaped

    def random_sample(self, batch):
        arr = []
        image_names = np.random.choice(self.arr, batch)
        for name in image_names:
            arr.append(self.read_image(name))
        image_array = np.array(arr)
        return image_array


