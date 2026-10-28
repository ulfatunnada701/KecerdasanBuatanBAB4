
# Example 4.20 - SimpleRNN3A.py
# Modified based on:
# https://www.datasciencenotes.com/2018/12/rnn-example-with-keras-simplernn-in.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Dense, SimpleRNN

# Convert into dataset matrix
def convertToMatrix(data, step):
    X, Y = [], []

    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d, ])
        Y.append(data[d, ])

    return np.array(X), np.array(Y)


# Load dataset
# https://www.kaggle.com/rakannimer/air-passengers

# df = pd.read_csv('AirPassengers.csv', usecols=[1], engine='python')

df = pd.read_csv(
    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',
    usecols=[1],
    engine='python'
)

print(df.head())

# Plot dataset
plt.plot(df)
plt.title('Air Passengers Dataset')
plt.show()

# Data preparation
step = 4
N = df.shape[0]
Tp = int(df.shape[0] * 0.8)

values = df.values
train, test = values[0:Tp, :], values[Tp:N, :]

# Add step elements into train and test
test = np.append(test, np.repeat(test[-1, ], step))
train = np.append(train, np.repeat(train[-1, ], step))

# Convert data
trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)

# Reshape data
trainX = np.reshape(
    trainX, (trainX.shape[0], 1, trainX.shape[1])
)

testX = np.reshape(
    testX, (testX.shape[0], 1, testX.shape[1])
)

# Create SimpleRNN model
model = Sequential()

model.add(Input(shape=(1, step)))
model.add(SimpleRNN(units=32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop'
)

model.summary()

# Train model
model.fit(
    trainX,
    trainY,
    epochs=10,
    batch_size=16,
    verbose=2
)

# Prediction
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

predicted = np.concatenate(
    (trainPredict, testPredict), axis=0
)

# Evaluation
trainScore = model.evaluate(trainX, trainY, verbose=0)
print("Training MSE:", trainScore)

# Plot results
index = df.index.values

plt.plot(df, label='Data')
plt.plot(predicted, label='Prediction')
plt.axvline(df.index[Tp], c='r')
plt.legend()
plt.title('SimpleRNN Air Passengers Prediction')
plt.xlabel('Time')
plt.ylabel('Number of Passengers')
plt.show()
