import numpy as np

# Linear sistem
A = np.array([[1, 2], [3, 4]])
B = np.array([1, 2])

print np.linalg.inv(A).dot(B)
print np.linalg.solve(A, B)

