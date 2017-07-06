import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from datetime import datetime
# from sklearn.utils import shuffle


def getData(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open('./fer.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y


def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y


def init_filter(shape, poolsz=(2,2)):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    # return tf.truncated_normal(shape, stddev=0.1)
    return w.astype(np.float32)

def init_weight(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32)

def init_bias(M2):
    b = np.zeros(M2)
    return b.astype(np.float32)

def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def accuracy_rate(targets, predictions):
    return np.mean(targets == predictions)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def main():
    X, Y = getImageData()
    # X, Y = shuffle(X, Y)
    X = X.astype(np.float32)
    Y = y2indicator(Y).astype(np.float32)
    # reshape X for tf: N x w x h x c
    X = X.transpose((0, 2, 3, 1))
    N, width, height, c = X.shape


    Xvalid, Yvalid = X[-1000:], Y[-1000:]
    X, Y = X[:-1000], Y[:-1000]
    Yvalid_flat = np.argmax(Yvalid, axis=1) # for calculating error rate

    # DATA = 0D 0-6 Emotion, 1D 48*48 Long Image, String Usage

    INPUTS = tf.placeholder(tf.float32, shape=(None, width, height, c))
    TARGETS = tf.placeholder(tf.float32, shape=(None, 7))

    x_image = tf.reshape(INPUTS, [-1, 48, 48, 1])

    # First convolution
    W_conv1 = tf.Variable(init_filter([5, 5, 1, 20]))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[20]))

    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # Reduced to 24*24

    #Second convolution
    W_conv2 = tf.Variable(init_filter([5, 5, 20, 20]))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[20]))

    h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # Reduced to 12*12

    # Flattern the image
    h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*20])

    # First Layer of the NN
    # W_fc1 = tf.Variable(tf.truncated_normal([12*12*20, 500], stddev=0.1))
    # b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
    W_fc1 = tf.Variable(init_weight(12*12*20, 500))
    b_fc1 = tf.Variable(init_bias(500))

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Second Layer of the NN
    # W_fc2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
    # b_fc2 = tf.Variable(tf.constant(0.1, shape=[300]))
    W_fc2 = tf.Variable(init_weight(500, 300))
    b_fc2 = tf.Variable(init_bias(300))

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2) # h_fc1_drop

    # Third Layer
    # W_fc3 = tf.Variable(tf.truncated_normal([300, 7], stddev=0.1))
    # b_fc3 = tf.Variable(tf.constant(0.1, shape=[7]))
    W_fc3 = tf.Variable(init_weight(300, 7))
    b_fc3 = tf.Variable(init_bias(7))

    OUTPUT = tf.matmul(h_fc2, W_fc3) + b_fc3

    prediction = tf.argmax(OUTPUT,1)

    # Calculate Cross-entropy error
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=TARGETS, logits=OUTPUT))

    # Train method
    train_step = tf.train.RMSPropOptimizer(10e-4, decay=0.99999, momentum=0.99).minimize(cross_entropy) # tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

    # Compare network output vs target
    correct_prediction = tf.equal(prediction, tf.argmax(TARGETS,1))
    # Cast bools to float32 and take the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    batch_sz = 30
    n_batches = N // batch_sz
    for i in range(3):
        t0 = datetime.now()
        for j in range(n_batches):
            Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
            Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]
            train_step.run(feed_dict={INPUTS: Xbatch, TARGETS: Ybatch, keep_prob: 0.75})
            if (j % 20 == 0):
                #train_accuracy = accuracy.eval(feed_dict={INPUTS: Xvalid, TARGETS: Yvalid, keep_prob: 1.0})
                p = session.run(prediction, feed_dict={INPUTS: Xvalid, TARGETS: Yvalid})
                e = error_rate(Yvalid_flat, p)
                a = accuracy_rate(Yvalid_flat, p)
                a2 = accuracy.eval(feed_dict={INPUTS: Xvalid, TARGETS: Yvalid, keep_prob: 1.0})
                print("step: ", i, " error rate: ", e, " accuracy rate 1: ", a, " accuracy rate 2: ", a2)
        dt1 = datetime.now() - t0
        print(dt1.total_seconds())


    # print("test accuracy %g"%accuracy.eval(feed_dict={INPUTS: mnist.test.images, TARGETS: mnist.test.labels, keep_prob: 1.0}))

if __name__ == "__main__":
    main()
