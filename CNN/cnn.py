import numpy as np
import pandas as pd
import tensorflow as tf
import random

class ConvLayer(object):
    def __init__(self, patch_x, patch_y, input_c, output_c):
        shape = [patch_x, patch_y, input_c, output_c]
        self.weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[output_c]))

    def forward(self, X, pool=False):
        convolution = tf.nn.conv2d(X, self.weights, strides=[1, 1, 1, 1], padding='SAME')
        convolution = tf.nn.bias_add(convolution, self.biases)
        if pool:
            p1, p2 = pool
            convolution = tf.nn.max_pool(convolution, ksize=[1, p1, p2, 1], strides=[1, p1, p2, 1], padding='SAME')
        return convolution


class HiddenLayer(object):
    def __init__(self, x, y):
        shape = [x, y]
        self.weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[y]))

    def forward(self, X):
        layer = tf.nn.relu(tf.matmul(X, self.weights) + self.biases)
        return layer


class CNN(object):
    def __init__(self, conv_layer_sizes, hidden_layer_sizes, output_shape):
        x, y = output_shape
        self.W = tf.Variable(tf.truncated_normal(output_shape, stddev=0.1))
        self.B = tf.Variable(tf.constant(0.1, shape=[y]))
        self.convpool_layers = []
        self.hidden_layers = []
        # [5, 5, 1, 20]
        # The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels.
        for patch_x, patch_y, input_c, output_c in conv_layer_sizes:
            layer = ConvLayer(patch_x, patch_y, input_c, output_c)
            self.convpool_layers.append(layer)

        for input_c, output_c in hidden_layer_sizes:
            layer = HiddenLayer(input_c, output_c)
            self.hidden_layers.append(layer)

    def salute(self):
        return "Hello!"

    def forward(self, X):
        Z = X
        for convolution in self.convpool_layers:
            Z = convolution.forward(Z, pool=[2, 2]) # Can add pooling here

        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])

        for hidden in self.hidden_layers:
            Z = hidden.forward(Z)

        Z = tf.matmul(Z, self.W) + self.B
        return Z

    def predict(self, X):
        Z = self.forward(X)
        prediction = tf.argmax(Z, 1)
        return prediction

    def trainRMSProp(self, data, epochs, batch_sz, learning_rate = 10e-4, decay = 0.99999, momentum = 0.99):
        pass

    def trainAdamOptimizer(self, data, epochs, batch_sz, learning_rate = 1e-2):
        pass

    def initTrainParams():
        pass

    def train(self, data, epochs, batch_sz, learning_rate = 10e-4, decay = 0.99999, momentum = 0.99):
        X, Y = data #getImageData()
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)
        K = len(set(Y))
        # reshape X for tf: N x w x h x c
        X = X.transpose((0, 2, 3, 1))
        N, width, height, c = X.shape

        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        Yvalid_flat = np.argmax(Yvalid, axis=1) # for calculating error rate

        INPUTS = tf.placeholder(tf.float32, shape=(None, width, height, c))
        TARGETS = tf.placeholder(tf.float32, shape=(None, K))

        OUTPUTS = self.forward(INPUTS)
        prediction = self.predic(INPUTS)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=TARGETS, logits=OUTPUTS))
        train_step = tf.train.RMSPropOptimizer(learning_rate, decay, momentum).minimize(cross_entropy) # tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

        # Compare network output vs target
        correct_prediction = tf.equal(prediction, tf.argmax(TARGETS,1))
        # Cast bools to float32 and take the mean
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        n_batches = N // batch_sz
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for e in range(epochs):
                for i in range(n_batches):
                    Xbatch = X[i*batch_sz:(i*batch_sz+batch_sz)]
                    Ybatch = Y[i*batch_sz:(i*batch_sz+batch_sz)]
                    train_step.run(feed_dict={INPUTS: Xbatch, TARGETS: Ybatch})
                    if (i % 20 == 0):
                        #train_accuracy = accuracy.eval(feed_dict={INPUTS: Xvalid, TARGETS: Yvalid, keep_prob: 1.0})
                        p = session.run(prediction, feed_dict={INPUTS: Xvalid, TARGETS: Yvalid})
                        error = np.mean(Yvalid_flat != p)
                        acc = accuracy.eval(feed_dict={INPUTS: Xvalid, TARGETS: Yvalid})
                        print("step: ", e, " error rate: ", error, " accuracy: ", acc)
