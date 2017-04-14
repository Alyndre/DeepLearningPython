import numpy as np

A = np.array([[1, 2], [3, 4]])
Ainv = np.linalg.inv(A)

print Ainv
print Ainv.dot(A)
print A.dot(Ainv)

print np.linalg.det(A)

print np.diag(A)

print np.diag([1, 2])

a = np.array([1, 2])
b = np.array([3, 4])

print np.outer(a, b)

print np.diag(A).sum()
print np.trace(A)

# Eigenvalues and Eigenvectors
X = np.random.randn(100, 3)  # 100 samples, 3 features
cov = np.cov(X.T)
print cov.shape
print cov

print np.linalg.eigh(cov)
print np.linalg.eig(cov)
