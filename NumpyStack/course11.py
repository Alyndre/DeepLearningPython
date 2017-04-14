import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("train.csv")

print df.shape

M = df.as_matrix()

im = M[0, 1:]  # SELECT 0 ROW AND GIVE ME EVERY COLUMN FROM 1 TO INF SINCE 0 IS LABEL
print im.shape

im = im.reshape(28, 28)

print im.shape

plt.imshow(im, cmap='gray')  # color map
plt.show()

plt.imshow(255-im, cmap='gray')  # color map
plt.show()

print M[0, 0]


