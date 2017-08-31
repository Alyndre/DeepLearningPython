import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts

from keras.models import Sequential
from keras.layers import Dense, Dropout
from util import getData, processData

np.random.seed(7)

if __name__ == "__main__":
    data = getData('2012-01-10', '2017-08-25')

    X_train, X_test, Y_train, Y_test = processData(data)

    # nn = NN([(39, 500)], [500, 1])
    # nn.train(X_train, Y_train, X_test, Y_test, 10000)

    model = Sequential()
    model.add(Dense(500, input_shape=(20,), init='uniform', activation='relu'))
    # model.add(Dense(500, init='uniform', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(250, init='uniform', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, init='uniform', activation='linear'))
    
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, Y_train, nb_epoch=550,  batch_size = 1, verbose=1, validation_split=0.1)
    score = model.evaluate(X_test, Y_test, batch_size=128)
    print(score)
