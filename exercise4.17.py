
# Exercise 4.17 - Visualize CNN Feature Maps

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from keras.applications.vgg19 import VGG19, preprocess_input
from keras.utils import load_img, img_to_array
from keras.models import Model

# Load VGG19 model
model = VGG19(weights='imagenet')

# Load image
img_path = Path(__file__).resolve().parent / 'Elephant.jpg'

img = load_img(img_path, target_size=(224, 224))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# Fungsi visualisasi feature maps
def plot_feature_maps(feature_maps, layer_name):
    n_filters = feature_maps.shape[-1]

    cols = 8
    rows = int(np.ceil(n_filters / cols))

    plt.figure(figsize=(14, rows * 1.8))

    for i in range(n_filters):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap='gray')
        plt.axis('off')

    plt.suptitle(f"Feature Maps - {layer_name}")
    plt.tight_layout()
    plt.show()


# Pilih empat convolutional layer pertama
layer_names = [
    'block1_conv1',
    'block1_conv2',
    'block2_conv1',
    'block2_conv2'
]

# Visualisasi setiap layer
for layer_name in layer_names:

    print(f"\n===== {layer_name} =====")

    feature_model = Model(
        inputs=model.inputs,
        outputs=model.get_layer(layer_name).output
    )

    feature_maps = feature_model.predict(img, verbose=0)

    print("Feature Maps Shape:", feature_maps.shape)

    plot_feature_maps(feature_maps, layer_name)
