
# Example 4.19 SimpleRNN2.py

from keras.models import Sequential
from keras.layers import Input, Dense, SimpleRNN, Activation
from keras import optimizers

# Create model
model = Sequential()

model.add(Input(shape=(49, 1)))
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(46))
model.add(Activation('softmax'))

# Compile model
adam = optimizers.Adam(learning_rate=0.001)

model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)

# Menampilkan struktur model
model.summary()
