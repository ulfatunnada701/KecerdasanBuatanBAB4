
# Exercise 4.16 - Visualize All CNN Filters

from keras.applications.vgg19 import VGG19
from matplotlib import pyplot as plt
import numpy as np

# Load VGG19 model
model = VGG19(weights='imagenet')

# Pilih convolutional layer pertama
layer = model.layers[1]

# Mengambil filter dan bias
filters, biases = layer.get_weights()

print("Layer:", layer.name)
print("Filter Shape:", filters.shape)
print("Bias Shape:", biases.shape)

# Normalisasi nilai filter
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min + 1e-8)

# Mengambil ukuran filter
s = filters.shape

# Menampilkan seluruh filter
n_filters, ix = s[3], 1

# Mengatur jumlah baris dan kolom
n_cols = 8
n_rows = int(np.ceil(n_filters / n_cols))

plt.figure(figsize=(14, 12))

for i in range(n_filters):

    # Mengambil filter ke-i
    f = filters[:, :, :, i]

    # Mengubah RGB filter untuk visualisasi
    plt.subplot(n_rows, n_cols, ix)
    plt.imshow(f)
    plt.title(f'Filter {i + 1}', fontsize=8)
    plt.axis('off')

    ix += 1

plt.suptitle("VGG19 - All Convolutional Filters")
plt.tight_layout()
plt.show()
