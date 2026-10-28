
# Exercise 4.4 - LeNet-5 Filter Comparison

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D

# Fungsi untuk membuat model LeNet-5
def create_model(filter1, filter2):
    model = Sequential()

    # Input Layer
    model.add(Input(shape=(32, 32, 1)))

    # Convolution Layer 1
    model.add(Conv2D(filters=filter1, kernel_size=(3, 3),
                     activation='relu'))

    # Average Pooling Layer 1
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Convolution Layer 2
    model.add(Conv2D(filters=filter2, kernel_size=(3, 3),
                     activation='relu'))

    # Average Pooling Layer 2
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Flatten Layer
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))

    # Output Layer
    model.add(Dense(units=10, activation='softmax'))

    return model


# Model Original
print("\n===== MODEL ORIGINAL (6, 16) =====")
model_original = create_model(6, 16)
model_original.summary()
params_original = model_original.count_params()

# Model Modifikasi 1
print("\n===== MODEL MODIFIKASI 1 (8, 32) =====")
model_mod1 = create_model(8, 32)
model_mod1.summary()
params_mod1 = model_mod1.count_params()

# Model Modifikasi 2
print("\n===== MODEL MODIFIKASI 2 (16, 32) =====")
model_mod2 = create_model(16, 32)
model_mod2.summary()
params_mod2 = model_mod2.count_params()

# Perbandingan Total Parameter
print("\n===== HASIL PERBANDINGAN =====")
print(f"Original (6, 16)  : {params_original:,} parameters")
print(f"Modifikasi (8, 32): {params_mod1:,} parameters")
print(f"Modifikasi (16,32): {params_mod2:,} parameters")
