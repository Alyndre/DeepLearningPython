import os
import numpy as np
import pandas as pd
import tensorflow as tf

def main():
    file = os.path.dirname(os.path.realpath(__file__)) + '\\fer.csv'
    d = pd.read_csv(file)
    # print(d.info())
    d_training = d[d['Usage'] == 'Training']
    d_test = d[d['Usage'] == 'PrivateTest']

    DATA_TEST = d_test.as_matrix()
    DATA = d_training.as_matrix()
    INPUTS = DATA[:, 1]
    TARGETS = DATA[:, 0]

    # DATA = 0D Emotion, 1D 48*48 Long Image, String Usage


    input_ph = tf.placeholder(tf.float32, shape=[None, 48*48])
    target_ph = tf.placeholder(tf.float32, shape=[None, 7])

if __name__ == "__main__":
    main()