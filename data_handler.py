import numpy as np


class Data_handler:

    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.y_0 = np.where(self.y == 0)
        self.y_1 = np.where(self.y == 1)
        self.y_2 = np.where(self.y == 2)
        self.y_3 = np.where(self.y == 3)
        self.y_4 = np.where(self.y == 4)
        self.y_5 = np.where(self.y == 5)
        self.y_6 = np.where(self.y == 6)
        self.y_7 = np.where(self.y == 7)
        self.y_8 = np.where(self.y == 8)
        self.y_9 = np.where(self.y == 9)

        self.train_x_0 = np.take(self.x, self.y_0, axis=0)[0]
        self.train_x_1 = np.take(self.x, self.y_1, axis=0)[0]
        self.train_x_2 = np.take(self.x, self.y_2, axis=0)[0]
        self.train_x_3 = np.take(self.x, self.y_3, axis=0)[0]
        self.train_x_4 = np.take(self.x, self.y_4, axis=0)[0]
        self.train_x_5 = np.take(self.x, self.y_5, axis=0)[0]
        self.train_x_6 = np.take(self.x, self.y_6, axis=0)[0]
        self.train_x_7 = np.take(self.x, self.y_7, axis=0)[0]
        self.train_x_8 = np.take(self.x, self.y_8, axis=0)[0]
        self.train_x_9 = np.take(self.x, self.y_9, axis=0)[0]

        self.random_sample(20)

    def random_sample(self, n):
        batch = int(n / 10)
        x_0 = np.reshape(self.train_x_0[np.random.choice(self.train_x_0.shape[0], batch, replace=False)],
                         (batch, 28, 28, 1))
        x_1 = np.reshape(self.train_x_1[np.random.choice(self.train_x_1.shape[0], batch, replace=False)],
                         (batch, 28, 28, 1))
        x_2 = np.reshape(self.train_x_2[np.random.choice(self.train_x_2.shape[0], batch, replace=False)],
                         (batch, 28, 28, 1))
        x_3 = np.reshape(self.train_x_3[np.random.choice(self.train_x_3.shape[0], batch, replace=False)],
                         (batch, 28, 28, 1))
        x_4 = np.reshape(self.train_x_4[np.random.choice(self.train_x_4.shape[0], batch, replace=False)],
                         (batch, 28, 28, 1))
        x_5 = np.reshape(self.train_x_5[np.random.choice(self.train_x_5.shape[0], batch, replace=False)],
                         (batch, 28, 28, 1))
        x_6 = np.reshape(self.train_x_6[np.random.choice(self.train_x_6.shape[0], batch, replace=False)],
                         (batch, 28, 28, 1))
        x_7 = np.reshape(self.train_x_7[np.random.choice(self.train_x_7.shape[0], batch, replace=False)],
                         (batch, 28, 28, 1))
        x_8 = np.reshape(self.train_x_8[np.random.choice(self.train_x_8.shape[0], batch, replace=False)],
                         (batch, 28, 28, 1))
        x_9 = np.reshape(self.train_x_9[np.random.choice(self.train_x_9.shape[0], batch, replace=False)],
                         (batch, 28, 28, 1))

        args = (x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9)
        arr = np.concatenate(args)
        np.random.shuffle(arr)
        return arr



