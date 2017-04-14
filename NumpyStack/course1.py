import numpy as np

L = [1, 2, 3]
A = np.array([1, 2, 3])

for e in L:
    print e

print "-------"

for e in A:
    print e

print "-------"

L.append(4)
print L

L2 = []

for e in L:
    L2.append(e + e)

print L2

print L + L2

print A + A
print 2*A
print 2*L

L2 = []

for e in L:
    L2.append(e*e)

print L2

print A**2
print np.sqrt(A)
print np.log(A)
print np.exp(A)

