
# Exercise 4.9 - VGG19

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions

# Load VGG19 model
model = VGG19(weights='imagenet')

# Menampilkan struktur model
model.summary()

# Menampilkan jumlah parameter
print("\n===== HASIL MODEL VGG19 =====")
print("Model: VGG19")
print(f"Total Parameters: {model.count_params():,}")
