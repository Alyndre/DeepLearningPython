import numpy as np
import pandas as pd
import os 

print("a")


# One Hot Encoding
def one_hot_encoding(Yin):
    N = Yin.size
    K = np.amax(Yin) + 1

    Yout = np.zeros((N, K.astype(np.int32)))

    for n in xrange(N):
        Yout[n, Yin[n].astype(np.int32)] = 1

    return Yout


def get_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(dir_path + '\ecommerce_data.csv', header=0).as_matrix()
    
    X = data[:, :-1]
    Y = data[:, -1]

    # normalize numerical columns -> (Value - Mean) / Standard Deviation
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

    N, D = X.shape  # Get X dimensions
    X2 = np.zeros((N, D+3))  # 3 More columns cause time_of_day would get One Hot Encoded and has 4 posible values
    X2[:,0:(D-1)] = X[:,0:(D-1)]  # Other columns are the same

    X2[:,-4:] = one_hot_encoding(X[:,4])

    return X2, Y


def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2

