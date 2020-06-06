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

        img = self.crop_around_center(image=img, width=120, height=120)
        resized = cv2.resize(img, (int(64), int(64)))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = (gray.astype(np.float32)) / 255
        gray_reshaped = np.reshape(gray, (64, 64, 1))
        return gray_reshaped

    def random_sample(self, batch):
        arr = []
        image_names = np.random.choice(self.arr, batch)
        for name in image_names:
            arr.append(self.read_image(name))
        image_array = np.array(arr)
        return image_array

    def crop_around_center(self, image, width, height):
        """
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around it's centre point
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if (width > image_size[0]):
            width = image_size[0]

        if (height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]


