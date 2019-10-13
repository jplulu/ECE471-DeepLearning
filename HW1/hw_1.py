# Junpeng Lu
# Prof. Curro
# CGML HW #1

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import trange

NUM_FEATURES = 4
NUM_SAMP = 50
BATCH_SIZE = 32
NUM_BATCHES = 300
LEARNING_RATE = 0.1


class Data(object):
    def __init__(self):
        num_samp = NUM_SAMP
        sigma = 0.1
        np.random.seed(31415)

        self.index = np.arange(num_samp)
        self.x = np.random.uniform(size=(num_samp, 1))
        self.y = np.sin(2 * np.pi * self.x) + sigma * np.random.normal()

    def get_batch(self, batch_size=BATCH_SIZE):
        choices = np.random.choice(self.index, size=batch_size)

        return self.x[choices].flatten(), self.y[choices].flatten()


class Model(tf.Module):
    def __init__(self, num_features=NUM_FEATURES):
        self.w = tf.Variable(tf.random.uniform(shape=[num_features, 1]))
        self.b = tf.Variable(tf.random.uniform(shape=[1, 1]))
        self.mu = tf.Variable(tf.random.uniform(shape=[num_features, 1]))
        self.sig = tf.Variable(tf.random.uniform(shape=[num_features, 1]))

    def __call__(self, x):
        return tf.squeeze(tf.transpose(self.w) @ tf.exp(-tf.square(x - self.mu) / (tf.square(self.sig))) + self.b)


if __name__ == "__main__":
    data = Data()
    model = Model()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch()
            y_hat = model(x)
            loss = tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    x_plt = np.linspace(0, 1, 100)
    y_plt = model(x_plt)

    plt.figure()
    plt.plot(x_plt, np.sin(2 * np.pi * x_plt), color='blue', linewidth=1)
    plt.plot(x_plt, y_plt, color='red', linestyle='dashed', linewidth=1)
    plt.scatter(data.x, data.y, color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fit 1')
    plt.show()

    plt.figure()
    for mu, sig in zip(model.mu.numpy(), model.sig.numpy()):
        plt.plot(x_plt, np.exp(-np.square(x_plt - mu) / (np.square(sig))))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bases for Fit 1')
    plt.show()
