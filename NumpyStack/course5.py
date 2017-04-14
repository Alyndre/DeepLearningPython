import numpy as np

# CREATING ARRAYS

A1 = np.array([1, 2, 3])

Z = np.zeros(10)
print Z

Z = np.zeros((10, 10))
print Z

O = np.ones((10, 10))
print O

R = np.random.random((10, 10))  # uniform distributed between 0 and 1
print R

G = np.random.randn(10, 10)  # Gaussian ditribution mean 0 variance 1
print G

print G.mean()
print G.var()
