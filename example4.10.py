
# Example 4.10 - VGG16

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np

# Load VGG16 model
model = VGG16(weights='imagenet')

# Menampilkan struktur model
model.summary()
