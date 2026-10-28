
# Exercise 4.6 - CNN Layer Comparison

from keras.models import Sequential
from keras.layers import (
    Input, Dense, Dropout, Flatten,
    Conv2D, MaxPooling2D
)

# Fungsi membuat model CNN
def create_model(modified=False):
    model = Sequential()

    model.add(Input(shape=(28, 28, 1)))

    # 1st Convolutional Layer
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Additional Layers (Exercise 4.6)
    if modified:
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


# Model Original (Example 4.7)
print("\n===== MODEL ORIGINAL =====")
model_original = create_model(modified=False)
model_original.summary()
params_original = model_original.count_params()

# Model Modifikasi (Exercise 4.6)
print("\n===== MODEL MODIFIKASI =====")
model_modified = create_model(modified=True)
model_modified.summary()
params_modified = model_modified.count_params()

# Perbandingan Parameter
print("\n===== HASIL PERBANDINGAN =====")
print(f"Original   : {params_original:,} parameters")
print(f"Modifikasi : {params_modified:,} parameters")

difference = params_original - params_modified
reduction = difference / params_original * 100

print(f"Selisih Parameter : {difference:,}")
print(f"Pengurangan Parameter : {reduction:.2f}%")
