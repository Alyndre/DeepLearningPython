import numpy as np
import pandas as pd
from datetime import datetime

X = []

for line in open("data_2d.csv"):
    row = line.split(',')
    sample = map(float, row)
    X.append(sample)

Y = np.array(X)
print Y
print Y.shape

X = pd.read_csv("data_2d.csv", header=None)
print type(X)

print X.info()
print X.head(5)


M = X.as_matrix()
print type(M)

print X[0]  # COLUMN NAME

# SERIES OBJECT, NOT NDARRAY
print X.iloc[0]
print X.ix[0]

print X[X[0] < 5]


df = pd.read_csv("international-airline-passengers.csv", engine="python", skipfooter=3)

print df.columns
df.columns = ["month", "passengers"]
print df.columns
print df['passengers']
print df.passengers
df['ones'] = 1
print df.head()


datetime.strptime("1949-05", "%Y-%m")

df["datetime"] = df.apply(lambda r: datetime.strptime(r['month'], "%Y-%m"), axis=1)
print df.info()


t1 = pd.read_csv('table1.csv')
t2 = pd.read_csv('table2.csv')

m = pd.merge(t1, t2, on='user_id')
print m
print t1.merge(t2, on='user_id')
