import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import datetime
import tensorflow as tf
from util import getData, processData

class HiddenLayer(object):
    def __init__(self, x, y):
        shape = [x, y]
        self.weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[y]))
        print(self.weights.shape)

    def forward(self, X):
        layer = tf.nn.relu(tf.matmul(X, self.weights) + self.biases)
        return layer

class NN(object):
    def __init__(self, hidden_layer_sizes, output_shape):
        x, y = output_shape
        self.W = tf.Variable(tf.truncated_normal(output_shape, stddev=0.1))
        self.B = tf.Variable(tf.constant(0.1, shape=[y]))
        self.hidden_layers = []

        for input_c, output_c in hidden_layer_sizes:
            layer = HiddenLayer(input_c, output_c)
            self.hidden_layers.append(layer)

    def forward(self, X):
        Z = X
        #Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])

        for hidden in self.hidden_layers:
            Z = hidden.forward(Z)

        Z = tf.matmul(Z, self.W) + self.B
        return Z

    def train(self, X, Y, Xt, Yt, epochs, batch_sz=1, learning_rate = 10e-4, decay = 0.99999, momentum = 0.99):
        N = X.shape[0]
        # reshape X for tf: N x w x h x c
        #X = np.transpose(X)
        print(X.shape)
        #N, width, height, c = X.shape

        INPUTS = tf.placeholder(tf.float32, shape=[None, X.shape[1]])
        TARGETS = tf.placeholder(tf.float32, shape=[None, Y.shape[1]])

        OUTPUTS = self.forward(INPUTS)

        loss = tf.reduce_sum(tf.squared_difference(TARGETS, OUTPUTS)) # tf.losses.mean_squared_error(labels=TARGETS, predictions=OUTPUTS)
        accuracy = tf.losses.mean_squared_error(labels=TARGETS, predictions=OUTPUTS)
        cost = tf.reduce_sum(tf.pow(OUTPUTS-TARGETS, 2))/(2*N)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # tf.train.RMSPropOptimizer(learning_rate, decay, momentum).minimize(accuracy) 
        # tf.train.AdamOptimizer(1e-2).minimize(accuracy) # 

        # Compare network output vs target
        correct_prediction = tf.equal(OUTPUTS, TARGETS)
        # Cast bools to float64 and take the mean

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            n_batches = N // batch_sz
            for e in range(epochs):
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                    Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]
                    train_step.run(feed_dict={INPUTS: Xbatch, TARGETS: Ybatch})
                if (e % 100 == 0):
                    #train_accuracy = accuracy.eval(feed_dict={INPUTS: Xvalid, TARGETS: Yvalid, keep_prob: 1.0})
                    p = session.run(OUTPUTS, feed_dict={INPUTS: Xt, TARGETS: Yt})
                    #error = np.mean(Yvalid_flat != p)
                    acc = accuracy.eval(feed_dict={OUTPUTS: p, TARGETS: Yt})
                    # l = tf.losses.mean_squared_error(feed_dict={INPUTS: X_test, TARGETS: Y_test})
                    print("step: ", e,  " accuracy: ", acc)

if __name__ == "__main__":
    data = getData('2012-01-10', '2017-08-25')

    X_train, X_test, Y_train, Y_test = processData(data)

    nn = NN([(39, 500)], [500, 1])
    nn.train(X_train, Y_train, X_test, Y_test, 10000)
