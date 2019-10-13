# Junpeng Lu
# CGML HW #3
# Prof. Curro

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import gzip

TRAINX_PATH = "train-images-idx3-ubyte.gz"
TRAINY_PATH = "train-labels-idx1-ubyte.gz"
TESTX_PATH = "t10k-images-idx3-ubyte.gz"
TESTY_PATH = "t10k-labels-idx1-ubyte.gz"
NUM_CLASSES = 10
IMG_ROWS = 28
IMG_COLUMNS = 28
VAL_PERC = 1 / 6
BATCH_SIZE = 256
L2_LAMBDA = 1e-5
EPOCHS = 10
np.random.seed(1234)


class Data(object):
    def __init__(self):
        f = gzip.open(TRAINX_PATH, 'rb')
        f.read(16)
        self.x_train = np.frombuffer(f.read(IMG_ROWS * IMG_COLUMNS * 60000), dtype=np.uint8).astype(np.float32)
        self.x_train = self.x_train.reshape(-1, IMG_ROWS, IMG_COLUMNS, 1)
        self.x_train = self.x_train / 255

        f = gzip.open(TRAINY_PATH, 'rb')
        f.read(8)
        self.y_train = np.frombuffer(f.read(60000), dtype=np.uint8).astype(np.float32)
        self.y_train = tf.keras.utils.to_categorical(self.y_train)

        f = gzip.open(TESTX_PATH, 'r')
        f.read(16)
        self.x_test = np.frombuffer(f.read(IMG_ROWS * IMG_COLUMNS * 10000), dtype=np.uint8).astype(np.float32)
        self.x_test = self.x_test.reshape(-1, IMG_ROWS, IMG_COLUMNS, 1)
        self.x_test = self.x_test / 255

        f = gzip.open(TESTY_PATH, 'rb')
        f.read(8)
        self.y_test = np.frombuffer(f.read(10000), dtype=np.uint8).astype(np.float32)
        self.y_test = tf.keras.utils.to_categorical(self.y_test)

        for _ in range(5):
            indexes = np.random.permutation(len(self.x_train))

        self.x_train = self.x_train[indexes]
        self.y_train = self.y_train[indexes]

        val_count = int(VAL_PERC * len(self.x_train))
        self.x_val = self.x_train[:val_count, :]
        self.y_val = self.y_train[:val_count, :]
        self.x_train = self.x_train[val_count:, :]
        self.y_train = self.y_train[val_count:, :]


if __name__ == "__main__":
    data = Data()
    x_train = data.x_train
    y_train = data.y_train
    x_test = data.x_test
    y_test = data.y_test
    x_val = data.x_val
    y_val = data.y_val
    input_shape = (IMG_ROWS, IMG_COLUMNS, 1)

    model = tf.keras.models.Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA)))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val), verbose=2)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
