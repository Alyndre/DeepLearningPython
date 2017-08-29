import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import datetime
import tensorflow as tf


def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def error_rate(targets, predictions):
    return np.mean(targets - predictions)


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
            newY.append([Y[index]])
            r = [*X[index-20: index], *Y[index-20: index-1]]
            newX.append(r)
    
    X_train = newX[0:len(newX) - 300]
    Y_train = newY[0:len(newY) - 300]
    
    X_train, Y_train = shuffle_in_unison(np.array(X_train).astype(np.float32), np.array(Y_train).astype(np.float32))

    X_test = newX[len(newX) - 300:]
    Y_test = newY[len(newY) - 300:]

    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

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