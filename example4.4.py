
# Example 4.4 LeNet-5 Keras

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D

# Membuat model Sequential
model = Sequential()

# Convolution Layer 1
model.add(Input(shape=(32, 32, 1)))
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))

# Pooling Layer 1
model.add(AveragePooling2D(pool_size=(2, 2)))

# Convolution Layer 2
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

# Pooling Layer 2
model.add(AveragePooling2D(pool_size=(2, 2)))

# Flatten Layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))

# Output Layer
model.add(Dense(units=10, activation='softmax'))

# Menampilkan struktur model
model.summary()
