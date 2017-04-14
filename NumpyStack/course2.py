import numpy as np

a = np.array([1, 2])
b = np.array([2, 1])

dot = 0

for e, f in zip(a, b):
    # e=a[0], f=b[0]
    dot += e*f

print dot

print a*b
print np.sum(a*b)

print np.dot(a, b)
print a.dot(b)
print a.dot(a)

# distance vector = arrel de (vector dot vector)
amag = np.sqrt(a.dot(a))
print amag
amag = np.linalg.norm(a)
print amag

cosangle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
print cosangle

angle = np.arccos(cosangle)
print angle
