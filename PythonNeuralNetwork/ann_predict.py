import numpy as np
from process import get_data

X, Y = get_data()

I = X.shape[1]  # Num of columns in X
H = 5
K = len(set(Y))

W1 = np.random.randn(I, H)
b1 = np.random.randn(H)
W2 = np.random.randn(H, K)
b2 = np.random.randn(K)

def forward(I, W1, b1, W2, b2):
    Z = tanh(I, W1, b1)
    A = Z.dot(W2) + b2
    return softmax(A, axis=1)

def classification_rate(Y, P):
    return np.mean(Y == P)

# Softmax
def softmax(a, axis=0):
    expa = np.exp(a)
    return expa / expa.sum(axis=axis, keepdims=True)

# Sigmoid
def sigmoid(M, W, b):
    return 1 / (1 + np.exp(-M.dot(W) - b))

# Hypertan
def tanh(X, W, b):
    return np.tanh(X.dot(W) + b)


P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)
print "Score:", classification_rate(Y, P)
