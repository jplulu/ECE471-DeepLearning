# Junpeng Lu
# CGML HW #2
# Prof. Curro

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

"""
I first created my model without any l2 regularization, the model was not able to learn at all. After adding l2
regularization, the model was able to perform better but still not ideal. Next, I increased the amount of neurons in
each layer as well as the number of batches, this greatly reduced the loss of the model, but it still was not modeling 
perfectly. After tweaking with the learning rate as well as the layer setup, the model is finally able to achieve 100%
accuracy. Lastly, I tried using elu instead of relu as the activation function, which seems to make the model slightly
better.
"""

NUM_CLASSES = 2
NUM_SAMP = 1000
BATCH_SIZE = 32
NUM_BATCHES = 2500
LEARNING_RATE = 0.1
LAMBDA = 0.01
LAYERS = [(2, 42), (42, 40), (40, 38), (38, 1)]


class Data(object):
    def __init__(self):
        num_samp = NUM_SAMP
        num_classes = NUM_CLASSES
        sigma = 0.1
        np.random.seed(31415)

        self.index = np.arange(num_samp * num_classes)
        self.x = np.zeros((num_samp * num_classes, 2), dtype='float32')
        self.y = np.zeros(num_samp * num_classes, dtype='float32')
        for c in range(num_classes):
            i = range(num_samp * c, num_samp * (c + 1))
            r = np.linspace(1, 15, num_samp)
            theta = (np.linspace(c * 3, (c + 4) * 3, num_samp) + np.random.randn(num_samp) * sigma)
            self.x[i] = np.c_[r * np.sin(theta), r * np.cos(theta)]
            self.y[i] = c

    def get_batch(self, batch_size=BATCH_SIZE):
        choices = np.random.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].reshape((batch_size, 1))


class Model(tf.Module):
    def __init__(self):
        self.weights = {}
        self.biases = {}
        for i in range(len(LAYERS)):
            nameW = f"w{i}"
            nameB = f"b{i}"
            self.weights[nameW] = tf.Variable(tf.random.normal(LAYERS[i]), name=nameW)
            self.biases[nameW] = tf.Variable(tf.zeros([LAYERS[i][1]]), name=nameB)

    def __call__(self, x):
        for weight in self.weights:
            if weight == "w0":
                self.y_hat = tf.matmul(x, self.weights[weight]) + self.biases[weight]
            else:
                self.y_hat = tf.matmul(tf.nn.elu(self.y_hat), self.weights[weight]) + self.biases[weight]
        return self.y_hat


if __name__ == "__main__":
    data = Data()
    model = Model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch()
            y_hat = model(x)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y))
            l2 = 0
            for weight in model.weights:
                l2 += tf.nn.l2_loss(model.weights[weight])
            loss = tf.reduce_mean(loss + LAMBDA * l2)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    n = 100
    x = np.linspace(-15, 15, n, dtype='float32')
    y = np.linspace(-15, 15, n, dtype='float32')
    xx, yy, = np.meshgrid(x, y)

    decision = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xy = np.reshape([x[j], y[i]], (1, 2))
            p = model(xy)
            if p >= .5:
                decision[i, j] = 1
    cm = plt.cm.RdBu

    plt.contourf(xx, yy, decision, cmap=cm, alpha=.5)
    plt.scatter(data.x[:, 0], data.x[:, 1], c=data.y,
                cmap=cm, alpha=1, edgecolors='black')
    plt.show()
