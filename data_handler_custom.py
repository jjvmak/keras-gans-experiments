import os
import cv2
import numpy as np


class Data_handler_custom:

    def __init__(self):
        self.arr = os.listdir(path='dataset')
        self.arr_len = len(self.arr)
        self.random_sample(10)

    def read_image(self, name):
        img = cv2.imread('./dataset/' + name)
        # img = img / 255
        resized = cv2.resize(img, (int(64), int(64)))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = gray / 255
        gray_reshaped = np.reshape(gray, (64, 64, 1))
        return gray_reshaped

    def random_sample(self, batch):
        arr = []
        image_names = np.random.choice(self.arr, batch)
        for name in image_names:
            arr.append(self.read_image(name))
        image_array = np.array(arr)
        return image_array


