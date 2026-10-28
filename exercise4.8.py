
# Exercise 4.8 - CNN MNIST Layer Comparison

import time
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (
    Input, Dense, Dropout, Flatten,
    Conv2D, MaxPooling2D
)
from keras.utils import to_categorical

# Load dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

# Normalize input from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

# Fungsi membuat model CNN
def create_model(modified=False):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))

    # 1st Convolutional Layer
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Additional Layers (Exercise 4.8)
    if modified:
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


# Fungsi training dan evaluasi
def train_model(modified=False):
    tf.keras.utils.set_random_seed(42)

    model = create_model(modified)

    name = "MODIFIKASI" if modified else "ORIGINAL"

    print(f"\n===== MODEL {name} =====")
    model.summary()

    # Menghitung training time
    start_time = time.perf_counter()

    model.fit(
        X_train, y_train,
        validation_split=0.1,
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

    print(f"\nHASIL MODEL {name}")
    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"CNN Error     : {error:.2f}%")
    print(f"Training Time : {training_time:.2f} detik")
    print(f"Total Params  : {parameters:,}")

    return accuracy, error, training_time, parameters


# Model Original (Example 4.9)
result_original = train_model(modified=False)

# Model Modifikasi (Exercise 4.8)
result_modified = train_model(modified=True)

# Perbandingan hasil
print("\n========== HASIL PERBANDINGAN ==========")

print("\nMODEL ORIGINAL")
print(f"Accuracy      : {result_original[0]:.2f}%")
print(f"CNN Error     : {result_original[1]:.2f}%")
print(f"Training Time : {result_original[2]:.2f} detik")
print(f"Total Params  : {result_original[3]:,}")

print("\nMODEL MODIFIKASI")
print(f"Accuracy      : {result_modified[0]:.2f}%")
print(f"CNN Error     : {result_modified[1]:.2f}%")
print(f"Training Time : {result_modified[2]:.2f} detik")
print(f"Total Params  : {result_modified[3]:,}")

print("\nSELISIH ACCURACY")
print(f"{result_modified[0] - result_original[0]:+.2f} poin persentase")
