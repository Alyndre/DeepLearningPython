from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt

print norm.pdf(0)
print norm.pdf(0, loc=5, scale=10)

r = np.random.randn(10)

print norm.pdf(r)
print norm.logpdf(r)

print norm.cdf(r)
print norm.logcdf(r)

r = np.random.randn(10000)

plt.hist(r, bins=100)
plt.show()

r = 10*np.random.randn(10000) + 5
plt.hist(r, bins=100)
plt.show()

r = np.random.randn(10000, 2)
plt.scatter(r[:,0], r[:,1])
plt.show()

r[:,1] = 5*r[:,1] +2

plt.scatter(r[:,0], r[:,1])
plt.axis('equal')
plt.show()

cov = np.array([[1, 0.8], [0.8, 3]])
mu = np.array([0, 2])
r = mvn.rvs(mean=mu, cov=cov, size=1000)
plt.scatter(r[:,0], r[:,1])
plt.axis('equal')
plt.show()

r = np.random.multivariate_normal(mean=mu, cov=cov, size=1000)
plt.scatter(r[:,0], r[:,1])
plt.show()

