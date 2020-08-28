import tensorflow as tf

loss_mapping = {"binary_cross_entropy": tf.keras.losses.binary_crossentropy,
                "categorical_crossentropy": tf.keras.losses.categorical_crossentropy, "mse": tf.keras.losses.MSE}
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from utils import random_mini_batches
from layers import Input, Dense
import time


def linear(x):
    return x


activation_mapping = {"sigmoid": tf.nn.sigmoid, "relu": tf.nn.relu, "tanh": tf.nn.tanh, "linear": linear,
                      "softmax": tf.nn.softmax}


class Model:
    def __init__(self, input=None, layers=None):
        if not input or not layers:
            raise ValueError("input and layers needed")
        else:
            if not isinstance(input, Input):
                raise ValueError("Input instance needed")
            self.input = input
            for layer in layers:
                if not isinstance(layer, Dense):
                    raise ValueError("Dense instance needed")
            self.layers = layers
            ops.reset_default_graph()
            self.initialize_parameter()

    def initialize_parameter(self):
        n_l_1 = self.input.shape

        for i in range(len(self.layers)):
            l = i + 1
            n_l = self.layers[i].units_num
            self.layers[i].setVariables("W" + str(l), "b" + str(l), n_l_1)

            n_l_1 = n_l

    def create_placeholders(self, n_x, n_y):

        X = tf.placeholder(tf.float32, [n_x, None])
        Y = tf.placeholder(tf.float32, [n_y, None])

        return X, Y

    def forward_propagation(self, X):
        A = X
        for i in range(len(self.layers)):
            l = i + 1
            layer = self.layers[i]
            W = layer.parameters["W"]
            b = layer.parameters["b"]

            Z = tf.matmul(W, A) + b
            if self.layers[i].activation == "softmax":
                A = tf.nn.softmax(Z, axis=0)
            else:
                A = activation_mapping[self.layers[i].activation](Z)
        return A

    def fit(self, X_train, Y_train, X_test, Y_test, loss="MSE", learning_rate=0.0001,
            num_epochs=5, minibatch_size=32, print_cost=True):

        tf.set_random_seed(1)
        seed = 3
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        if n_x != self.input.shape:
            raise IndexError("input of shape (" + str(self.input.shape) + ", ) is needed")
        X, Y = self.create_placeholders(n_x, n_y)

        Y_hat = self.forward_propagation(X)
        cost = tf.reduce_mean(loss_mapping[loss](tf.transpose(Y), tf.transpose(Y_hat)))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        costs = []
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                st = time.time()
                epoch_cost = 0.  # Defines a cost related to an epoch
                num_minibatches = int(
                    m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                    epoch_cost += minibatch_cost / num_minibatches

                    # Print the cost every epoch
                if print_cost == True:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)
                    # Calculate the correct predictions
                    # plot the cost
                print(time.time() - st)

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(Y_hat), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

