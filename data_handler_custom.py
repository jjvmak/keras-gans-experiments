import os
import cv2
import numpy as np

# Data handler for LFW dataset.
# TODO: make this more generic for other dataset as well.
class Data_handler_custom:

    def __init__(self):
        # Load the filenames from the disk.
        self.arr = os.listdir(path='dataset')

    # Reads single image from the disk.
    # Crop 120x120 area from the center of the image.
    # Resize to 64x64.
    # Transform to grayscale.
    # Normalize pixel values in range 0 to 1.
    # Reshape to (64, 64, 1)
    def read_image(self, name):
        """
            Args:
                name: image name

            Returns:
                gray_reshaped: cropped and reshaped gray scale image

        """
        img = cv2.imread('./dataset/' + name)
        img = self.crop_around_center(image=img, width=120, height=120)
        resized = cv2.resize(img, (int(64), int(64)))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = (gray.astype(np.float32)) / 255
        gray_reshaped = np.reshape(gray, (64, 64, 1))
        return gray_reshaped

    # Draws a random sample size of batch.
    def random_sample(self, batch):
        """
            Args:
                batch: size of batch

            Returns:
                image_array: array of images size of batch

        """
        arr = []
        image_names = np.random.choice(self.arr, batch)
        for name in image_names:
            arr.append(self.read_image(name))
        image_array = np.array(arr)
        return image_array

    #   point.
    def crop_around_center(self, image, width, height):
        """Given a NumPy / OpenCV 2 image,
           crops it to the given width and height, around it's centre

           Args:
               image: image
               width: desired width
               height desired height

            Returns:
                image: cropped image

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


