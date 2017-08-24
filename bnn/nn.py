import requests
import numpy as np
import tensorflow as tf

def getData():
    r = requests.get('https://api.coindesk.com/v1/bpi/historical/close.json?currency=EUR')
    print(r.json())

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(lr = 1e-4):
    I = 784
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

    LAYER1 = tf.nn.relu(tf.matmul(INPUT, WEIGHT1) + BIAS1)
    LAYER2 = tf.nn.relu(tf.matmul(LAYER1, WEIGHT2) + BIAS2)
    OUTPUT = tf.matmul(LAYER2, WEIGHT3) + BIAS3

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=TARGET, logits=OUTPUT))
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(TARGET,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={INPUT:batch[0], TARGET: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={INPUT: batch[0], TARGET: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={INPUT: mnist.test.images, TARGET: mnist.test.labels, keep_prob: 1.0}))


if __name__ == "__main__":
    main()