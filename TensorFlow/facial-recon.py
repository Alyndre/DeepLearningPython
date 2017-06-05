import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from datetime import datetime
from sklearn.utils import shuffle


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


def main():
    file = os.path.dirname(os.path.realpath(__file__)) + '\\fer.csv'
    d = pd.read_csv(file)
    # print(d.info())
    d_training = d[d['Usage'] == 'Training']
    d_test = d[d['Usage'] == 'PrivateTest']


    # TensorFlow Session
    session = tf.InteractiveSession()


    DATA_TEST = d_test.as_matrix()
    DATA = d_training.as_matrix()
    inputs_data = DATA[:, 1]
    # One Hot Encoding
    targets_data = tf.one_hot(DATA[:, 0], 7).eval()
    # inputs_data, targets_data = shuffle(inputs_data, targets_data)
    inputs_data = [i.split(' ') for i in inputs_data]


    # Make batches from the data
    n = 20
    i_batches = np.array([inputs_data[i:i + n] for i in range(0, len(inputs_data), n)])
    t_batches = np.array([targets_data[i:i + n] for i in range(0, len(targets_data), n)])

    # DATA = 0D 0-6 Emotion, 1D 48*48 Long Image, String Usage

    INPUTS = tf.placeholder(tf.float32, shape=[None, 48*48])
    TARGETS = tf.placeholder(tf.float32, shape=[None, 7])

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

    # Calculate Cross-entropy error
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=TARGETS, logits=OUTPUT))

    # Train method
    train_step = tf.train.RMSPropOptimizer(10e-4, decay=0.99999, momentum=0.99).minimize(cross_entropy) # tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

    # Compare network output vs target
    correct_prediction = tf.equal(tf.argmax(OUTPUT,1), tf.argmax(TARGETS,1))

    # Cast bools to float32 and take the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session.run(tf.global_variables_initializer())

    for i in range(3):
        x = random.choice(range(len(i_batches)))
        train_accuracy = accuracy.eval(feed_dict={INPUTS: i_batches[x], TARGETS: t_batches[x], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        t0 = datetime.now()
        for l in range(len(i_batches)):
            i_batch = i_batches[l]
            t_batch = t_batches[l]
            train_step.run(feed_dict={INPUTS: i_batch, TARGETS: t_batch, keep_prob: 1.0})
            if (l % 20 == 0):
                train_accuracy = accuracy.eval(feed_dict={INPUTS: t_batches[x], TARGETS: t_batches[x], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))

        dt1 = datetime.now() - t0
        print(dt1.total_seconds())


    # print("test accuracy %g"%accuracy.eval(feed_dict={INPUTS: mnist.test.images, TARGETS: mnist.test.labels, keep_prob: 1.0}))

if __name__ == "__main__":
    main()
