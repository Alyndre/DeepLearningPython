import numpy as np

# X1 = children, X2 = adults
# x1 + x2 = 2200
# 1.5x1 + 4x2 = 5050
# (  1, 1) (X1) = (2200)
# (1.5, 4) (X2) = (5050)

A = np.array([[1, 1], [1.5, 4]])
B = np.array([2200, 5050])
print np.linalg.solve(A, B)

