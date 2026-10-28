
import time
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
whole_data = load_breast_cancer()
x_data = whole_data.data
y_data = whole_data.target

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=7
)

features = X_train.shape[1]

# Fungsi untuk membuat dan menguji model
def train_model(neurons):
    tf.keras.utils.set_random_seed(42)

    # MLP 3-layer Model
    model = Sequential()
    model.add(Input(shape=(features,)))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1))

    print(f"\nMODEL DENGAN {neurons} NEURON")
    model.summary()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mse']
    )

    # Mengukur waktu training
    start_time = time.perf_counter()

    model.fit(
        X_train, y_train,
        batch_size=50,
        validation_split=0.2,
        epochs=100,
        verbose=0
    )

    training_time = time.perf_counter() - start_time

    # Evaluasi model
    results = model.evaluate(X_test, y_test, verbose=0)

    # Menghitung accuracy
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = (predictions >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, predicted_classes)

    print("Loss:", results[0])
    print("MSE:", results[1])
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Training Time: {training_time:.2f} detik")

    return training_time, accuracy

# Model awal: 10 neuron
time_10, acc_10 = train_model(10)

# Model modifikasi: 5 neuron
time_5, acc_5 = train_model(5)

# Perbandingan hasil
print("\n===== PERBANDINGAN MODEL =====")
print(f"10 Neuron: {time_10:.2f} detik | Accuracy: {acc_10*100:.2f}%")
print(f"5 Neuron : {time_5:.2f} detik | Accuracy: {acc_5*100:.2f}%")
