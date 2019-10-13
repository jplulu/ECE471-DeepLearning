# Junpeng Lu
# CGML HW #4
# Prof. Curro

"""
I tried to use the network I used for the Mnist data set, this worked very poorly. Next, I attempted to implement
ResNet based on the google paper. However, the model did not work very well, it trained very slow since I could only
use tensorflow cpu, it also did not converge very well, getting stuck around 80%. Next, I decided to go back to using
convolutional neural networks. After some research, I found that augmenting the input data using the image data
generator. In addition, I increase the depth and width of the network and added batch normalization to every
layer in order to avoid overfitting. This model trained up to around 85% accuracy. Next, I decided to changed the 
optimizer to RMSProp over ADAM and added decaying learning rate schedule over epochs. This network was surprisingly 
able to achieve 88-89% accuracy over 125 epochs. The network was also used to classify cifar100 data set, achieving 
top 5 85% accuracy over 125 epochs.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.datasets import cifar10, cifar100
import matplotlib.pyplot as plt

NUM_CLASSES = 100  # 10 for cifar10
IMG_ROWS = 32
IMG_COLUMNS = 32
IMG_DEPTH = 3
VAL_PERC = 1 / 5
BATCH_SIZE = 64
baseMapNum = 32
weight_decay = 1e-4
EPOCHS = 25
np.random.seed(1234)


class Data(object):
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()  # or cifar10
        self.x_train = self.x_train.astype('float32')
        self.x_train = self.x_train.reshape(-1, IMG_ROWS, IMG_COLUMNS, IMG_DEPTH)
        self.x_test = self.x_test.astype('float32')
        self.x_test = self.x_test.reshape(-1, IMG_ROWS, IMG_COLUMNS, IMG_DEPTH)

        mean = np.mean(self.x_train, axis=(0, 1, 2, 3))
        std = np.std(self.x_train, axis=(0, 1, 2, 3))
        self.x_train = (self.x_train - mean) / (std + 1e-7)
        self.x_test = (self.x_test - mean) / (std + 1e-7)

        self.y_train = tf.keras.utils.to_categorical(self.y_train, NUM_CLASSES)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, NUM_CLASSES)

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

    # Based on keras website
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)
    datagen.fit(x_train)

    input_shape = (IMG_ROWS, IMG_COLUMNS, IMG_DEPTH)

    model = tf.keras.models.Sequential()
    model.add(Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    opt_rms = optimizers.RMSprop(lr=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt_rms,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=x_train.shape[0] // BATCH_SIZE, epochs=3 * EPOCHS, verbose=1,
                        validation_data=(x_val, y_val))
    model.save_weights('cifar100_normal_rms_ep75.h5')

    opt_rms = optimizers.RMSprop(lr=0.0005, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt_rms,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=x_train.shape[0] // BATCH_SIZE, epochs=EPOCHS, verbose=2,
                        validation_data=(x_val, y_val))
    model.save_weights('cifar100_normal_rms_ep100.h5')

    opt_rms = optimizers.RMSprop(lr=0.0003, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt_rms,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=x_train.shape[0] // BATCH_SIZE, epochs=EPOCHS, verbose=2,
                        validation_data=(x_val, y_val))

    print(model.evaluate(x_test, y_test, batch_size=128, verbose=0))
