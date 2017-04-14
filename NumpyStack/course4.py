import numpy as np

M = np.array([[1, 2], [3, 4]])
L = [[1, 2], [3, 4]]

print L[0]
print L[0][0]

print M[0][0]
print M[0, 0]

M2 = np.matrix([[1, 2], [3, 4]])

# np.array > np.matrix

A = np.array(M2)

print A.T  # Matrix transpose
