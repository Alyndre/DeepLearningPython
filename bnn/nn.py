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
        N =1
        # reshape X for tf: N x w x h x c
        #X = X.transpose((0, 2, 3, 1))
        #N, width, height, c = X.shape

        INPUTS = tf.placeholder(tf.float32)
        TARGETS = tf.placeholder(tf.float32)

        OUTPUTS = self.forward(INPUTS)
        prediction = self.forward(INPUTS)

        loss = tf.reduce_sum(tf.squared_difference(TARGETS, OUTPUTS)) # tf.losses.mean_squared_error(labels=TARGETS, predictions=OUTPUTS)
        train_step = tf.train.AdamOptimizer(1e-2).minimize(loss) # tf.train.RMSPropOptimizer(learning_rate, decay, momentum).minimize(loss)

        # Compare network output vs target
        correct_prediction = tf.equal(prediction, TARGETS)
        # Cast bools to float64 and take the mean
        accuracy = tf.losses.mean_squared_error(labels=TARGETS, predictions=OUTPUTS)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for e in range(epochs):
                for (Xbatch, Ybatch) in zip(X, Y):
                    print(Xbatch)
                    print(Ybatch)
                    train_step.run(feed_dict={INPUTS: Xbatch, TARGETS: Ybatch})
                    if (i % 20 == 0):
                        #train_accuracy = accuracy.eval(feed_dict={INPUTS: Xvalid, TARGETS: Yvalid, keep_prob: 1.0})
                        p = session.run(prediction, feed_dict={INPUTS: Xbatch, TARGETS: Ybatch})
                        #error = np.mean(Yvalid_flat != p)
                        acc = accuracy.eval(feed_dict={INPUTS: Xbatch, TARGETS: Ybatch})
                        # l = tf.losses.mean_squared_error(feed_dict={INPUTS: X_test, TARGETS: Y_test})
                        print("step: ", e, "prediction: ", p, "loss: ", loss, " accuracy: ", acc)

if __name__ == "__main__":
    data = getData('2012-01-10', '2017-08-25')

    X_train, X_test, Y_train, Y_test = processData(data)

    print(X_train.shape)
    print(Y_train.shape)

    nn = NN([(39, 500), (500, 300)], [300, 1])

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    v = sess.run(nn.forward(X_train))    
    #print(v)

    nn.train(X_train, Y_train, X_test, Y_test, 50)
