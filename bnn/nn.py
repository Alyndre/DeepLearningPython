import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import datetime
import tensorflow as tf

def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def create_Xt_Yt(X, Y):
    newX = []# np.ndarray(shape=len(X)-20, 21)
    newY = [] # np.ndarray(shape=len(X)-20, 1)

    for index in range(len(X)):
        if index > 20:
            newY.append(Y[index])
            r = [*X[index-20: index], *Y[index-20: index-1]]
            newX.append(r)

    
    print(newX[len(newX)-1])
    
    X_train = X[0:len(X) - 300]
    Y_train = Y[0:len(Y) - 300]
    
    X_train, Y_train = shuffle_in_unison(np.array(X_train), np.array(Y_train))

    X_test = X[len(X) - 300:]
    Y_test = Y[len(X) - 300:]

    return X_train, X_test, Y_train, Y_test

def getData(start, end):
    r = requests.get('https://api.coindesk.com/v1/bpi/historical/close.json?start=' + start +'&end=' + end)
    print('https://api.coindesk.com/v1/bpi/historical/close.json?start=' + start +'&end=' + end);
    return r.json()["bpi"]

def processData(data):
    lists = sorted(data.items())

    x, y = zip(*lists)
    dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in x]
    dates = dts.date2num(dates)
    
    return create_Xt_Yt(dates, y)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(X_train, X_test, Y_train, Y_test, lr = 1e-4):
    I = 2*20
    L1 = 500
    L2 = 300
    O = 1

    INPUT = tf.placeholder(tf.float32, [None, I])
    TARGET = tf.placeholder(tf.float32, [None, O])

    WEIGHT1 = weight_variable([I, L1])
    BIAS1 = bias_variable([L1])

    WEIGHT2 = weight_variable([L1, L2])
    BIAS2 = bias_variable([L2])

    WEIGHT3 = weight_variable([L2, O])
    BIAS3 = bias_variable([O])

    DROPOUT = tf.placeholder(tf.float32)
    LAYER1 = tf.nn.relu(tf.matmul(INPUT, WEIGHT1) + BIAS1)
    LAYER1_DROP = tf.nn.dropout(LAYER1, DROPOUT)
    LAYER2 = tf.nn.relu(tf.matmul(LAYER1_DROP, WEIGHT2) + BIAS2)
    LAYER2_DROP = tf.nn.dropout(LAYER2, DROPOUT)
    OUTPUT = tf.matmul(LAYER2_DROP, WEIGHT3) + BIAS3

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=TARGET, logits=OUTPUT))
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy) # tf.train.RMSPropOptimizer(10e-4, decay=0.99999, momentum=0.99).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):

        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={INPUT: input_data, TARGET: target_data, DROPOUT: 1.0})

    # Test trained model
    correct_prediction = tf.equal(OUTPUT, TARGET)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={INPUT: X_test, TARGET: Y_test, DROPOUT: 1.0}))


if __name__ == "__main__":
    data = getData('2012-01-10', '2017-08-25')

    X_train, X_test, Y_train, Y_test = processData(data)

    # main(X_train, X_test, Y_train, Y_test)
    
