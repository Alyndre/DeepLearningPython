import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main():
    mnist = input_data.read_data_sets('/mnist', one_hot=True)

    I = 784
    L1 = 300
    O = 10

    INPUT = tf.placeholder(tf.float32, [None, I])
    TARGET = tf.placeholder(tf.float32, [None, O])

    WEIGHT1 = weight_variable([I, L1])
    BIAS1 = bias_variable([L1])
    WEIGHT2 = weight_variable([L1, O])
    BIAS2 = bias_variable([O])

    LAYER1 = tf.nn.relu(tf.matmul(INPUT, WEIGHT1) + BIAS1)
    OUTPUT = tf.matmul(LAYER1, WEIGHT2) + BIAS2


    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=TARGET, logits=OUTPUT))
    train_step = tf.train.MomentumOptimizer(0.75, 0.1).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={INPUT: batch_xs, TARGET: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(OUTPUT, 1), tf.argmax(TARGET, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={INPUT: mnist.test.images, TARGET: mnist.test.labels}))


if __name__ == "__main__":
    main()
