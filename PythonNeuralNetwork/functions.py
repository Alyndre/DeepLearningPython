import numpy as np

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

Y = np.array([1, 1, 0])
print convert_numbered_targets_to_indicator_matrix(Y)

a = np.random.randn(5)
exa = softmax(a)
print exa
print exa.sum()

A = np.random.randn(100, 5)
exA = softmax(A, axis=1)
print exA
print exA.shape
print exA.sum(axis=1)
