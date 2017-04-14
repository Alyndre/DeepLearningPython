import numpy as np
import matplotlib.pyplot as plt

data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
targetPos = np.argmax(target, axis=1)

D = 2  # Inputs
M = 4  # Hidden
K = 2  # Outputs

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def forward(I, W1, b1, W2, b2):
    Z = sigmoid(I, W1, b1)
    A = Z.dot(W2) + b2
    return softmax(A, axis=1)

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in xrange(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

# Softmax
def softmax(a, axis=0):
    expa = np.exp(a)
    return expa / expa.sum(axis=axis, keepdims=True)

# Sigmoid
def sigmoid(M, W, b):
    return 1 / (1 + np.exp(-M.dot(W) - b))

# One Hot Encoding
def convert_numbered_targets_to_indicator_matrix(Yin):
    N = Yin.size
    K = np.amax(Yin) + 1

    Yout = np.zeros((N, K))

    for n in xrange(N):
        Yout[n, Yin[n]] = 1

    return Yout


P_Y_given_X = forward(data, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)
print "Classification rate for randomly chosen weights:", classification_rate(targetPos, P)

