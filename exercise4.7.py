
# Exercise 4.7 - CNN MNIST Comparison

import time
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (
    Input, Dense, Dropout, Flatten,
    Conv2D, MaxPooling2D
)
from keras.utils import to_categorical

# Random seed agar pengujian lebih konsisten
tf.keras.utils.set_random_seed(42)

# Load dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encode
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

# Fungsi untuk membuat dan menguji model CNN
def train_model(filters, kernel_size):
    tf.keras.utils.set_random_seed(42)

    model = Sequential()

    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters, kernel_size,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print(f"\n===== FILTER {filters}, KERNEL {kernel_size} =====")
    model.summary()

    # Training time
    start_time = time.perf_counter()

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=200,
        verbose=1
    )

    training_time = time.perf_counter() - start_time

    # Evaluation
    scores = model.evaluate(X_test, y_test, verbose=0)

    accuracy = scores[1] * 100
    error = 100 - accuracy
    parameters = model.count_params()

    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"CNN Error     : {error:.2f}%")
    print(f"Training Time : {training_time:.2f} seconds")
    print(f"Total Params  : {parameters:,}")

    return accuracy, error, training_time, parameters


# Model Original
result1 = train_model(32, (5, 5))

# Modifikasi 1: Filter 32 menjadi 64
result2 = train_model(64, (5, 5))

# Modifikasi 2: Kernel (5,5) menjadi (3,3)
result3 = train_model(64, (3, 3))

# Perbandingan hasil
print("\n========== HASIL PERBANDINGAN ==========")

results = [
    ("Original (32, 5x5)", result1),
    ("Filter 64 (64, 5x5)", result2),
    ("Kernel 3x3 (64, 3x3)", result3)
]

for name, result in results:
    print(f"\n{name}")
    print(f"Accuracy      : {result[0]:.2f}%")
    print(f"CNN Error     : {result[1]:.2f}%")
    print(f"Training Time : {result[2]:.2f} detik")
    print(f"Total Params  : {result[3]:,}")
