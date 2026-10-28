
# Example 4.3 Multiple Layer Perceptron Classification

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
whole_data = load_breast_cancer()
x_data = whole_data.data
y_data = whole_data.target

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=7
)

features = X_train.shape[1]

# MLP 3-layer Model
model = Sequential()
model.add(Dense(10, activation='relu',
                input_shape=(features,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# Menampilkan struktur model
print(model.summary())
print(model.get_config())

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mse']
)

# Training model
model.fit(
    X_train, y_train,
    batch_size=50,
    validation_split=0.2,
    epochs=100,
    verbose=1
)

# Evaluasi model
results = model.evaluate(X_test, y_test)

print('loss:', results[0])
print('accuracy:', results[1])
