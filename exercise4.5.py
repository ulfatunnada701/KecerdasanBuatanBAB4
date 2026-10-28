
# Exercise 4.5 - AlexNet Dense Layer Comparison

from keras.models import Sequential
from keras.layers import (
    Input, Dense, Dropout, Flatten,
    Conv2D, MaxPooling2D
)

# Fungsi membuat model AlexNet
def create_model(dense1, dense2, dense3):
    model = Sequential()

    model.add(Input(shape=(224, 224, 3)))

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, kernel_size=(11, 11),
                     activation='relu', strides=(4, 4),
                     padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     activation='relu', padding='same'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     activation='relu', padding='same'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(dense1, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(dense2, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(dense3, activation='relu'))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(17, activation='softmax'))

    return model


# Model Original
print("\n===== MODEL ORIGINAL =====")
model_original = create_model(4096, 4096, 1000)
model_original.summary()
params_original = model_original.count_params()

# Model Modifikasi
print("\n===== MODEL MODIFIKASI =====")
model_modified = create_model(1024, 512, 256)
model_modified.summary()
params_modified = model_modified.count_params()

# Hasil Perbandingan
print("\n===== HASIL PERBANDINGAN =====")
print(f"Original   : {params_original:,} parameters")
print(f"Modifikasi : {params_modified:,} parameters")

reduction = (params_original - params_modified) / params_original * 100

print(f"Pengurangan Parameter: {reduction:.2f}%")
