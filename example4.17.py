
# Example 4.17 - CNN Visualize Filters
# Part 1 - Part 6

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from keras.applications.vgg19 import VGG19, preprocess_input
from keras.utils import load_img, img_to_array
from keras.models import Model

# ===================================
# PART 1 - Load VGG19 Model
# ===================================

model = VGG19(weights='imagenet')
model.summary()

# ===================================
# PART 2 - List All Layers
# ===================================

print("\n===== ALL LAYERS =====")

for n, layer in enumerate(model.layers):
    print(n, layer.name)

# ===================================
# PART 3 - Convolutional Filters
# ===================================

print("\n===== CONVOLUTIONAL FILTERS =====")

for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape, biases.shape)

# ===================================
# PART 4 - Visualize Filters
# ===================================

n = 1  # First convolutional layer
filters, biases = model.layers[n].get_weights()

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min + 1e-8)

s = filters.shape
n_filters, ix = s[3], 1

cols = 8
rows = int(np.ceil(n_filters / cols))

plt.figure(figsize=(12, 12))

for i in range(n_filters):
    f = filters[:, :, :, i]

    plt.subplot(rows, cols, ix)
    plt.imshow(f)
    plt.axis('off')
    ix += 1

plt.suptitle("VGG19 Convolutional Filters")
plt.tight_layout()
plt.show()

# ===================================
# PART 5 - Plot Feature Maps Function
# ===================================

def plot_feature_maps(feature_maps):
    # Plot all feature maps
    col = 8
    n_maps = feature_maps.shape[3]
    row = int(np.ceil(n_maps / col))
    ix = 1

    plt.figure(figsize=(16, 16))

    for i in range(n_maps):
        ax = plt.subplot(row, col, ix)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(
            feature_maps[0, :, :, i],
            cmap='gray'
        )

        ix += 1

    plt.suptitle("VGG19 Feature Maps")
    plt.tight_layout()
    plt.show()

# ===================================
# PART 6 - Visualize Feature Maps
# ===================================

# Select hidden layer
n = 1

# Create feature extraction model
feature_model = Model(
    inputs=model.inputs,
    outputs=model.layers[n].output
)

feature_model.summary()

# Load image
img_path = Path(__file__).resolve().parent / 'Elephant.jpg'

img = load_img(img_path, target_size=(224, 224))
img = img_to_array(img)

# Expand dimensions
img = np.expand_dims(img, axis=0)

# Preprocess image
img = preprocess_input(img)

# Get feature maps
feature_maps = feature_model.predict(img, verbose=0)

print("Feature maps:", feature_maps.shape)

# Visualize feature maps
plot_feature_maps(feature_maps)
