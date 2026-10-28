
# Exercise 4.15 - Visualize Second Convolutional Layer

from keras.applications.vgg19 import VGG19
from matplotlib import pyplot as plt
import numpy as np

# Load VGG19 model
model = VGG19(weights='imagenet')

# Pilih convolutional layer kedua
n = 2

# Ambil filter dari layer yang dipilih
filters, biases = model.layers[n].get_weights()

print("Layer:", model.layers[n].name)
print("Filter Shape:", filters.shape)
print("Bias Shape:", biases.shape)

# Normalisasi filter untuk visualisasi
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min + 1e-8)

# Menampilkan 6 filter pertama
n_filters = 6

plt.figure(figsize=(10, 6))

for i in range(n_filters):
    # Filter pada input channel pertama
    f = filters[:, :, 0, i]

    plt.subplot(2, 3, i + 1)
    plt.imshow(f, cmap='gray')
    plt.title(f'Filter {i + 1}')
    plt.axis('off')

plt.suptitle('VGG19 - Second Convolutional Layer')
plt.tight_layout()
plt.show()
