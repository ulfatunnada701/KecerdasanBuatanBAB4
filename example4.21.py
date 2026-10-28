
# Example 4.21 SimpleLSTM.py
# Based on:
# https://www.tensorflow.org/guide/keras/rnn

from tensorflow import keras
from tensorflow.keras import layers

# Create model
model = keras.Sequential()

model.add(layers.Embedding(input_dim=100, output_dim=32))
model.add(layers.LSTM(64))
model.add(layers.Dense(1))

# Build model agar jumlah parameter langsung ditampilkan
model.build(input_shape=(None, 10))

# Menampilkan struktur model
model.summary()
